import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from mpl_toolkits.basemap import Basemap
import xarray as xr
import matplotlib as mpl
import os
from pydantic import BaseModel
# from pathlib import Path
from typing import List, Dict
from tqdm import tqdm 

# DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

class Data(BaseModel):
    xco2: List
    latitude: List
    longitude: List
    time: List

class Datas(BaseModel): 
    oco2_data_list: List[Data] = []
    # xco2: List 
    # latitude: List
    # longitude: List
    # time: List
    
    @classmethod
    def read_from_nc4(cls, fn_list: List[str]):
        """read the netcdf4 data
    
        Args:
            fn_list: List that contains paths to the OCO2 data file

        returns: 
            xco2: XCO2 data
            latitude, longitude: location 
            time: date & time when OCO2 observation is recorded (which is not used when visualizing the data)
        """
        oco2_data_list = []
        for fn in tqdm(fn_list): 
            try:
                with xr.open_dataset(fn) as ds:
                    xco2 = ds.xco2.values.tolist()
                    latitude = ds.latitude.values.tolist()
                    longitude = ds.longitude.values.tolist()
                    time = ds.time.values.tolist()

                    oco2_data_list.append(Data( 
                        xco2 = xco2, 
                        latitude = latitude, 
                        longitude = longitude, 
                        time = time, 
                    ))
            except ValueError: 
                # raise ValueError("Error in reading data.")
                pass 

        return cls(oco2_data_list=oco2_data_list)



class Map(): 
    def __init__(self,
            path_dir: str, 
            time_start: str, 
            time_last: str,
            xco2_min : float = 380, 
            xco2_max : float = 435,
            fov_lon: List =[139.3, 140.4], 
            fov_lat: List =[35.15, 36.05],
            show_flist = False, 
            map_name = 'tokyo', 
            ) -> None:
            
        """ arguments for plotting the figure
        """
        self.time_start = time_start 
        self.time_last = time_last 
        self.xco2_min = xco2_min 
        self.xco2_max = xco2_max 
        self.fov_lon = fov_lon 
        self.fov_lat = fov_lat 
        self.show_flist = show_flist
        self.map_name = map_name

        self.time_start2 = ''

        """ read the data
        """
        flist, date_infor = self.__get_filelist(path_dir, time_start, time_last)
        self.flist = flist
        self.date_infor = date_infor  
        if show_flist: 
            print(flist)
        self.oco2_data_list = Datas.read_from_nc4(flist).oco2_data_list

        """ variables for storing the necessary data in the plotting
        """
        self.useful_file = []


    def __get_filelist(self, path_dir: str, time_start: str, time_last: str) -> (List, List): 
        """ fetch the files (filenames) under a certain folder, with time time within the specified time window
        Args: 
            path_dir: path to the folder where data is placed (e.g., path_dir = '../../data/data_xco2/oco2_LtCO2_*.nc4')
            time_start: start time for the visualizing duration
            time_last:  end time for the visualizing duration

        Returns: 
            flist_all: all the filepaths 
            date_infor_all: date stamp of all the files (datetime.datetime(2014, 9, 6, 0, 0)) 
        
        """
        if path_dir[-1] != '/': 
            path_dir = path_dir + '/'
        flist2 = np.array(glob(path_dir+'*.nc4'))
        date_infor = np.array([ datetime.strptime('20'+ os.path.split(x)[1][-37:-31], '%Y%m%d')  for x in flist2])
        
        flist_all = flist2[np.argsort(date_infor)]
        date_infor_all = np.sort(date_infor)

        self.__validate(time_start)
        self.__validate(time_last)

        start_time = datetime.strptime(time_start, '%Y-%m-%d')
        end_time   = datetime.strptime(time_last , '%Y-%m-%d')
        need_ind  = np.where((date_infor_all <= end_time) * (date_infor_all >=start_time ))[0]
        
        flist = [ flist_all[ind] for ind in need_ind ]
        date_infor = [date_infor_all[ind] for ind in need_ind ]

        return flist, date_infor

    def __validate(self, date_text: str):
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    def __time_format_transform(self, datetime_string: str) -> str: 
        datetime_stamp = datetime.strptime(datetime_string, '%Y-%m-%d')
        datetime_string = datetime.strftime(datetime_stamp, '%Y%m%d')
        return datetime_string

    def __great_circle(self, lon1, lat1, lon2, lat2):
        """calculate the great circle distance in km
        
        Args: 
            lon1, lat1: location at point1 
            lon2, lat2: location at point2 
        
        Returns: 
            great distance between point1 and point2 in km
        """
        
        lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
        return 6371 * (
            np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
        )

    def __get_useful_data(self, oco2_data_list: List[Data]) -> (np.array, np.array, np.array, np.array, List, List): 
        
        """excludnig the loaded data that are out of the specified FOV.
        Args: 
            oco2_data_list: oco2 data object list

        """
        useful_file = []
        useful_date_infor = []
        useful_xco2 = np.array([])
        useful_latitude = np.array([])
        useful_longitude = np.array([])
        useful_time = np.array([])     

        for i, oco2_data in enumerate(oco2_data_list): 
            latitude = np.array(oco2_data.latitude) 
            longitude = np.array(oco2_data.longitude) 
            fov_ind = (longitude>=self.fov_lon[0]) * (longitude<=self.fov_lon[1]) * (latitude>=self.fov_lat[0]) * (latitude<=self.fov_lat[1])
            
            if np.sum(fov_ind) > 0: 
                ind = np.where(fov_ind)[0]
                useful_xco2 =  np.append( useful_xco2, np.array(oco2_data.xco2)[ind]) 
                useful_time = np.append(useful_time, np.array(oco2_data.time)[ind]) 
                useful_latitude = np.append(useful_latitude, latitude[ind] ) 
                useful_longitude = np.append( useful_longitude, longitude[ind]) 
                useful_file.append(self.flist[i]) 
                useful_date_infor.append(self.date_infor[i])

        return useful_xco2, useful_time, useful_latitude, useful_longitude, useful_file, useful_date_infor

    def set_useful_data(self,): 

        if self.useful_file == []: 
            useful_xco2, useful_time, useful_latitude, useful_longitude, useful_file, useful_date_infor = self.__get_useful_data(self.oco2_data_list)

            # set up the data that would be used in the plotting: 
            # these data are with the data points that falling in the specified fov
            self.useful_file = useful_file 
            self.useful_xco2 = useful_xco2
            self.useful_latitude = useful_latitude
            self.useful_longitude = useful_longitude
            self.useful_time = useful_time
            self.useful_date_infor = useful_date_infor



    def plot_fig_check_value_range(self, mode_2panel=False): 
        
        """ generating a figure for confirming the data range during the year.
        """    
        if self.useful_file == []: 
            self.set_useful_data()
            
        plt.clf()
        plt.close()
        ax = plt.subplot()


        # legend_timestamp = [ datetime.strftime(date_infor,'%Y-%m-%d') for date_infor in self.useful_date_infor ] 

        if len(self.time_start2) == 0 or not mode_2panel : 
            ax.plot(self.useful_time, 
                        self.useful_xco2, 
                        marker='o', 
                        markersize=4 , 
                        linestyle='none',
                        alpha=0.2, 
                    )
        else: 
            ax.plot(self.useful_time, 
                self.useful_xco2, 
                marker='o', 
                markersize=4 , 
                linestyle='none', 
                label='duration1', 
                alpha=0.2
            )
            ax.plot(self.useful_time2, 
                self.useful_xco22, 
                marker='o', 
                markersize=4 , 
                linestyle='none', 
                label='duration2', 
                alpha=0.2
            )
                    
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.axhline(y=self.xco2_min, color='k', lw=0.5, ls=':' )
        ax.axhline(y=self.xco2_max, color='k', lw=0.5, ls=':')

        ax.set_xlabel('time')
        ax.set_ylabel('xco2')

        ax.set_title('Data collected during {} - {}'.format(self.time_start, self.time_last))

        plt.tight_layout()

        time_start2 = self.__time_format_transform(self.time_start)
        time_last2 = self.__time_format_transform(self.time_last)

        figname = 'fig/xcoc2-{}_{}-{}.png'.format(time_start2,time_last2,self.map_name)
        plt.savefig(figname, dpi=250)

        print('{} is generated.'.format(figname))


    def plot_dotplot_fig(self, jump=100 ): 
        
        """ Generating a figure for visualizing 2D distribution of OCO2 observation by simply making a dot plot.
        
        Args: 
            jump: only pick up data at a 100-datapoint span (i.e., the 0th, 100th, 200th ... data point) so the the plotting process would not be too costy. 
        """    

        plt.close()
        plt.clf()
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)

        # load the data
        if self.useful_file == []: 
            self.set_useful_data()
        xco2 = self.useful_xco2
        longitude = self.useful_longitude 
        latitude = self.useful_latitude
        time = self.useful_time
        
        # color map information
        N = 255 ; cmap = plt.get_cmap('jet', N)

        color_value = (xco2-self.xco2_min)*(254.0/(self.xco2_max-self.xco2_min))
        for i in np.arange(0, len(xco2), jump) :
            v = color_value[i]
            lon = longitude[i]
            lat = latitude[i]
            ax.plot(lon, lat, c=cmap(int(v)), marker='o')
            
        ax.set_xlim(self.fov_lon)
        ax.set_ylim(self.fov_lat)
        ax.set_xlabel('Longitude ($^o$)')
        ax.set_ylabel('Latitude ($^o$)')
        ax.set_title('Data collected during {} - {}'.format(self.time_start, self.time_last))

        norm = mpl.colors.Normalize(vmin=self.xco2_min, vmax=self.xco2_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.set_ylabel('XCO2 (ppm)')

        plt.tight_layout()

        time_start2 = self.__time_format_transform(self.time_start)
        time_last2 = self.__time_format_transform(self.time_last)
        figname = 'fig/xco2_point-{}_{}-{}.png'.format(time_start2,time_last2,self.map_name)
        plt.savefig(figname, dpi=250)

        print('{} is generated.'.format(figname))


    def __make_meshgrid(self, useful_xco2, grid_num,
                        useful_longitude, useful_latitude, 
                        xco2_min, xco2_max, 
                        fov_lon, fov_lat ) -> (np.array, np.array, np.array,): 
        
        """Mapping the Xco2 observation into a mesh grid array. Values are scaled into (0, 155).  
        Grid with multiple data points in it are filled with the averaged values over the data.
        Args: 
            useful_xco2: xco2 data points observed within in the specified FOV. 
            useful_longitude, useful_latitude: corresponding lon and lat of xco2 data points.
            grid_num: the number that slicing latitude and longitude.
            [xco2_min, xco2_max]: value range in y-axis. 
            fov_lon, fov_lat: FOV of the map. By default FOV is within a [139.3, 140.4,35.15, 36.05] window

        returns: 
            x, y: x- and y-axis of the mesh grid map.
            z_map: resulting map    
        
        """


        lat_slice = np.linspace(fov_lat[0], fov_lat[1], grid_num)
        lon_slice = np.linspace(fov_lon[0], fov_lon[1], grid_num)
        x, y = np.meshgrid(lon_slice, lat_slice)
        z  = np.empty(x.flatten().shape) 
        z[:] = np.nan


        # colar value rangin from 0-254
        # rescale the xco2 values within this range 
        color_value = (useful_xco2-xco2_min)*(254.0/(xco2_max-xco2_min))
        # calcuate great circle distance of every point to the meshgrids
        # find the meshgrid point that is with the shortest distance 
        for i in range(len(useful_xco2)): 
            gcd = self.__great_circle(useful_longitude[i], useful_latitude[i], 
                                        x.flatten(), y.flatten())
            gcd_arg = gcd.argmin()
            if np.isnan(z[gcd_arg]) :
                z[gcd_arg] = color_value[i]
            else: 
                # take average if there has been a data located already
                z[gcd_arg] = np.mean( [color_value[i] ,z[gcd_arg]] )  
                
        # reshape the flattenized array 
        z_map = z.reshape(x.shape)

        return x, y, z_map


    def plot_meshgridmap_fig(self, grid_num=int(200/5) ): 
        
        """basically share the same function with `fig_point_plot` but in a mesh grid manner. 
        Grid with multiple data points in it are filled with an averaged value.
        
        Args:
            grid_num: a number denoting how many slices that latitude and longitude should be divided into.
        """
        # load the data
        if self.useful_file == []: 
            self.set_useful_data()

        # xco2 = self.useful_xco2
        # longitude = self.useful_longitude 
        # latitude = self.useful_latitude
        # time = self.useful_time

        x, y, z_map = self.__make_meshgrid(
                                self.useful_xco2, grid_num,
                                self.useful_longitude, self.useful_latitude, 
                                self.xco2_min, self.xco2_max, 
                                self.fov_lon, self.fov_lat,)
        
        plt.clf()
        plt.close()

        fig, ax = plt.subplots(figsize=(12,12))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=1.3)
        m = Basemap(llcrnrlon=x.min(), llcrnrlat=y.min(), urcrnrlon=x.max(), urcrnrlat=y.max(), \
                    rsphere=(6378137.00,6356752.3142),\
                    resolution='h',projection='merc', ax=ax)

        m.drawcoastlines(color='gray',zorder=5)
        m.drawparallels(np.arange(10,90,0.5),labels=[1,1,0,1], zorder=5)
        m.drawmeridians(np.arange(120,160,0.5),labels=[1,1,0,1], zorder=5)
        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')

        x2 = np.linspace(0, m.urcrnrx, z_map.shape[1])
        y2 = np.linspace(0, m.urcrnry, z_map.shape[0])

        xx, yy = np.meshgrid(x2, y2)
        cs = m.pcolormesh(xx, yy, z_map, cmap='jet', zorder=10, alpha=0.5,vmin=0, vmax=254)
        cbar = plt.colorbar(cs, cax=cax)

        # Avoid strips appearing in the colorbar
        cbar.set_alpha(1)
        cbar.draw_all()
        
        # # adjust the cax tick labels 
        cax_tick_label = cax.get_yticks()
        # cax.set_yticklabels(np.linspace(xco2_min, xco2_max, len(cax_tick_label)))
        cax.set_yticklabels(['{0:.0f}'.format(x)  for x in np.linspace(self.xco2_min, self.xco2_max, len(cax_tick_label))])
        cax.set_ylabel('XCO2 (ppm)', fontsize=20)

        # time_yy = os.path.split(useful_file[0])[1][-37:-31]
        grid_size = round(100/z_map.shape[0])
        time_start2 = self.__time_format_transform(self.time_start)
        time_last2 = self.__time_format_transform(self.time_last)   
        ax.set_title('Data collected during {}-{}'.format(self.time_start, self.time_last))
        figname = 'fig/xcoc2_map_meshgrid-{}_{}-{}-{}.png'.format(time_start2, time_last2, grid_size, self.map_name)

        plt.tight_layout()
        plt.savefig(figname, dpi=250)
        print('{} is generated.'.format(figname))


    def add_map(self, path_dir2, time_start2, time_last2 ): 
                
        """ arguments for plotting the figure
        """
        self.time_start2 = time_start2
        self.time_last2 = time_last2 

        """ read the data
        """
        flist2, date_infor2 = self.__get_filelist(path_dir2, time_start2, time_last2)
        self.flist2 = flist2
        self.date_infor2 = date_infor2  
        if self.show_flist: 
            print(flist2)
        self.oco2_data_list2 = Datas.read_from_nc4(flist2).oco2_data_list

        """ variables for storing the necessary data in the plotting
        """
        self.useful_file2 = []
        self.set_useful_data2()


    def set_useful_data2(self):

        if len(self.time_start2) == 0: 
            print('Conduct add_map method first')
        else: 
            if self.useful_file2 == []: 
                useful_xco22, useful_time2, useful_latitude2, useful_longitude2, useful_file2, useful_date_infor2 = self.__get_useful_data(self.oco2_data_list2)

                # set up the data that would be used in the plotting: 
                # these data are with the data points that falling in the specified fov
                self.useful_file2 = useful_file2
                self.useful_xco22 = useful_xco22
                self.useful_latitude2 = useful_latitude2
                self.useful_longitude2 = useful_longitude2
                self.useful_time2 = useful_time2
                self.useful_date_infor2 = useful_date_infor2

    def __fig_map_single(self, ax, fov_lon, fov_lat): 

        """plot a base world map on the spacified axes object.
        Args: 
            ax: 
            fov_lon, fov_lat: FOV 
        
        Returns:
            m: basemap object 

        """

        m = Basemap(llcrnrlon=fov_lon[0], llcrnrlat=fov_lat[0], urcrnrlon=fov_lon[1], urcrnrlat=fov_lat[1],\
                    rsphere=(6378137.00,6356752.3142),\
                    resolution='h',projection='merc', ax=ax)

        m.drawcoastlines(color='gray',zorder=5)
        m.drawparallels(np.arange(10,90,0.5),labels=[1,1,0,1], zorder=5)
        m.drawmeridians(np.arange(120,160,0.5),labels=[1,1,0,1], zorder=5)
        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')

        # x_tok, y_tok = m(Tok_lon,Tok_lat )
        # m.plot(x_tok, y_tok, '^r', markersize=10)
        return m
    
    def __cbar(self, cax,cs, xco2_min, xco2_max):
        """ add color bar on the figure 
        """
        # # color bar is only located next to the second axis 
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=1.3)
        cbar = plt.colorbar(cs, cax=cax, orientation='horizontal', ticks=np.linspace(0, 254, 5)) 
        cbar.set_alpha(1)
        cbar.draw_all()     # Avoid stange strips appearing in the colorbar
        # cax_tick_label = cbar.ax.get_xticks()     # # adjust the cax tick labels 
        # cbar.ax.set_xticks(np.linspace(0, 254, 5))
        cbar.ax.set_xticklabels(['{0:.0f}'.format(x)  for x in np.linspace(xco2_min, xco2_max, 5)])
        cax.set_xlabel('XCO2 (ppm)', fontsize=20)


    def plot_meshgridmap_2panel_fig(self, grid_num=int(200/5)): 

        """basically share the same function with `fig_point_plot` but in a mesh grid manner. 
        Grid with multiple data points in it are filled with an averaged value.
        
        Args:
            grid_num: a number denoting how many slices that latitude and longitude should be divided into.
        """


        self.set_useful_data2()

        plt.clf()
        plt.close()

        x, y, z_map1 = self.__make_meshgrid(
                                self.useful_xco2, grid_num,
                                self.useful_longitude, self.useful_latitude, 
                                self.xco2_min, self.xco2_max, 
                                self.fov_lon, self.fov_lat,)
        x, y, z_map2 = self.__make_meshgrid(
                                self.useful_xco22, grid_num,
                                self.useful_longitude2, self.useful_latitude2, 
                                self.xco2_min, self.xco2_max, 
                                self.fov_lon, self.fov_lat,)

        # split the subplots 
        fig, axs = plt.subplots(1,2, figsize=(19,12))

        # plot each subplot
        for ax, z_map in  zip(axs, [z_map1, z_map2] ): 
            m = self.__fig_map_single(ax, fov_lon=self.fov_lon, fov_lat=self.fov_lat)

            x2 = np.linspace(0, m.urcrnrx, z_map.shape[1])
            y2 = np.linspace(0, m.urcrnry, z_map.shape[0])
            xx, yy = np.meshgrid(x2, y2)
            cs = m.pcolormesh(xx, yy, z_map, cmap='jet', zorder=10, alpha=0.5,vmin=0, vmax=254)

            grid_size = round(200/z_map.shape[0]) # fov is 200X200 km
        
        # add color bar
        cax = fig.add_axes([0.2, 0.08, 0.5, 0.03]) 
        self.__cbar(cax,cs, self.xco2_min, self.xco2_max)

        # transform the datetime stamp string format
        time1_start2 = self.__time_format_transform(self.time_start)
        time1_last2 = self.__time_format_transform(self.time_last)
        time2_start2 = self.__time_format_transform(self.time_start2)
        time2_last2 = self.__time_format_transform(self.time_last2)

        # add titles
        axs[0].set_title('20{}-20{}'.format(time1_start2, time1_last2), fontsize=20    )
        axs[1].set_title('20{}-20{}'.format(time2_start2, time2_last2), fontsize=20    )
            
        # final 
        plt.tight_layout()

        figname = 'fig/xcoc2_map_meshgrid-{}_{}-{}_{}-{}.png'.format(time1_start2, time1_last2, time2_start2, time1_last2, grid_size,)
        plt.savefig(figname, dpi=250)
        print('{} is generated.'.format(figname))

    def set_co2_minmax(self, xco2_min: float, xco2_max: float ): 
        self.xco2_min = xco2_min
        self.xco2_max = xco2_max
        print(f'chnage the visializing range from {xco2_min} to {xco2_max}')

    # def set_fov(self, fov ): 
    #     self.fov_lon[0]= fov[0]
    #     self.fov_lon[1]= fov[1]
    #     self.fov_lat[0]= fov[2]
    #     self.fov_lat[1]= fov[3]
                

