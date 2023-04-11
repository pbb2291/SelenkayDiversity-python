
# New Complexity Function for lablidar_functions
# Copy of Function used in Selenkay Diversity Project 
# PB 10/31/22
# NOTE: If performing over a set of pixels, need to wrap the below in a loop
def canopyLayerMetrics(h, hmax, step, groundthres, hmin=0, plot=False, smoothsigma=2):
    # Where h is an array of point height values
    # hbins is a list/array of height bin edges
    # smoothsigma = positive float - gives a smoothing parameter for gaussian smoothing in layer calculation (in meters)
    # plot = True/False - whether to plot the results
    
    # If there are any heights in the array
    # edited 10/26 - have to have at least 2 points above the groundthreshold to do complexity stats
    # This means that a plot with only 1 point will have all metrics automatically set to 0 or nan
    if np.array(h).size >= 2:

        # Calc Cover for height bins
        nbins = ((hmax - hmin) / step) + 1
        hbins = np.linspace(hmin, hmax, int(nbins))

        # IMPORTANT: Using groundthres, you may want to account for errors in relative accuracy
        # EX: IF the rel. accuracy of ground is about 0.06 m (6 cm) between flightlines,
        # the lowest height bin could be set to 0.06 (instead of 0) to account for this.
        # so any hit below 0.06 m counts as ground.
        # NOTE: If you want to use everything, just set groundthres to 0
        if groundthres > 0:
            # insert the groundthres into the array (right above 0)
            heightbins = np.insert(hbins, 1, groundthres)
            # heightbins[heightbins==0] = groundthres
        if groundthres < 0:
            # insert the groundthres into the array (right below 0)
            heightbins = np.insert(hbins, 0, groundthres)
            # heightbins[heightbins==0] = groundthres

        # sort points by height
        h = np.sort(h)

        # Group each point by height bin
        hgroups = np.digitize(h, bins=hbins)

        # Count the number of points in each bin
        # Note: minlength allows it to go to the full length of hbins, rather than stop at the max height of the points
        hcounts = np.bincount(hgroups, minlength=len(hbins))

        # Normalize the counts
        hcounts_norm = hcounts/np.sum(hcounts)

        # Smooth point height distribution, and find peaks and troughs\
        # Following the below article on stack exchange
        # https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

        # smooth with gaussian filter
        smooth = gaussian_filter1d(hcounts_norm, smoothsigma)

        # compute first derivative
        smooth_d1 = np.gradient(smooth)
        
        # Interpolate heights and derivative to 1 cm increments so that you have more precise inflection points
        hbins_interp= np.arange(0, np.max(hbins), 0.01)
        smooth_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth)
        smooth_d1_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth_d1)

        # find inflection points
        # when it switches from positive to negative (a trough)
        idx_troughs = np.where(np.diff(np.sign(smooth_d1_interp))>0)[0]
        # when it switches from negative to positive (a peak)
        idx_peaks = np.where(np.diff(np.sign(smooth_d1_interp))<0)[0]

        # output height values of peaks and troughs
        troughs = hbins_interp[idx_troughs]
        peaks = hbins_interp[idx_peaks]

        # Number of layers as number of peaks
        nlayers = len(peaks)
        
        if nlayers > 1:
            
            # height location of peak with largest frequency of points
            maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            
            # Get the gap size as the maximum distance between consecutive peaks
            gapsize = np.max(np.diff(peaks))
            # gapsize = gapsize.round(3)
            
            # MEAN and STD peak HEIGHT
            # made up by peter
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nanstd(peaks)
            cvpeakh = np.nanmean(peaks)/np.nanstd(peaks)
            
            # Round
            # stdpeakh = stdpeakh.round(3)
            # cvpeakh = cvpeakh.round(3)
            
            # Vertical Distribution Ratio (Goetz 2007) 
            # - experimental, computed from peaks instead of norm point distribution
            VDRpeak = (np.max(peaks) - np.nanmedian(peaks)) / np.max(peaks)
            # VDRpeak = np.round(VDRpeak, 3)
            
            
        else:
            
            # Else: set to nan, so that only pixels that
            # have multiple layers in them are sampled
            # try:
            #     maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            #     meanpeakh = np.nanmean(peaks)
            # except:
            #     maxpeakh = np.nan
            #     meanpeakh = np.nan
            
            # Edited 10/31/22 
            # If there's only 1 layer, 
            # set the peaks to be the only peak height
            if peaks.size:
                maxpeakh = peaks[0]
                meanpeakh = peaks[0]
            # Except if there is no peak to record
            else:
                maxpeakh = 0
                meanpeakh = 0
                
            # and variation metrics to 0
            stdpeakh = 0
            gapsize = 0
            cvpeakh = 0
            VDRpeak = 0

        # computing the PtoH ratio (Davies 2020, Asner 2014)
        perc99 = np.nanpercentile(h, 99, method='median_unbiased')
        ptoh = maxpeakh/perc99
        # ptoh = ptoh.round(3)

        # Complexity Score (Davies 2020)
        # proportion of bins with points in them vs without
        # could be insensitive to striping
        cscore = np.sum(hcounts>0)/len(hcounts)
        # cscore = cscore.round(3)

        # Vertical Distribution Ratio (Goetz 2007)
        VDR = (np.max(h) - np.median(h)) / np.max(h)
        # VDR = np.round(VDR, 3)

        # Foliage Height Diversity 
        # (Bergen 2009 and many others using PAI profile - this is using normalized point counts)
        if np.sum(hcounts_norm>0) > 0:
            FHD = -1*np.sum(hcounts_norm*np.log(hcounts_norm, where=hcounts_norm>0)) 
            # FHD = FHD.round(3)
        else:
            FHD = 0

        ### SAVE Outputs
        complex_dict = {'nlayers':nlayers,
                        'gapsize':gapsize,
                        'maxpeakh':maxpeakh,
                        'ptoh':ptoh,
                        'cscore':cscore,
                        'FHD':FHD,
                        'VDR':VDR,
                        'VDRpeak':VDRpeak,
                        'meanpeakh':meanpeakh,
                        'stdpeakh':stdpeakh,
                        'cvpeakh':cvpeakh
                       }

        if plot:

            fig, ax = plt.subplots()
            ax.plot(hcounts_norm, hbins, label='Raw', lw=2, alpha=0.6)
            ax.plot(smooth, hbins, label='Smoothed', lw=2)
            # for infl in troughs:
            #     tline = ax.axhline(y=infl, color='b', label='Trough', alpha=0.6)
            for infl in peaks:
                pline = ax.axhline(y=infl, color='c', label='Peak', alpha=0.6)
            # ax.legend(handles=[tline, pline], loc='best')
            ax.legend(loc='best')
            ax.set_xlabel('Normalized Frequency of Points')
            ax.set_ylabel('Height [m]')
            ax.set_xlim(-0.01, np.max(hcounts_norm) + 0.03)
            ax.set_ylim(0, np.max(hbins) + 0.3)

            # don't return peaks, troughs
            return complex_dict, fig, ax

        else:

            return complex_dict
    
    # Else, if there were no heights in the array
    # just return an array of 0s
    else:
        
        complex_dict = {'nlayers':0,
                        'gapsize':0,
                        'maxpeakh':0,
                        'ptoh':0,
                        'cscore':0,
                        'FHD':0,
                        'VDR':0,
                        'VDRpeak':0,
                        'meanpeakh':0,
                        'stdpeakh':0,
                        'cvpeakh':0
                       }
        
        return complex_dict


def canopyLayerMetrics(h, hmax, step, groundthres, hmin=0, plot=False, smoothsigma=2):
    # Where h is an array of point height values
    # hbins is a list/array of height bin edges
    # smoothsigma = positive float - gives a smoothing parameter for gaussian smoothing in layer calculation (in meters)
    # plot = True/False - whether to plot the results
    
    # If there are any heights in the array
    # edited 10/26 - have to have at least 2 points above the groundthreshold to do complexity stats
    # This means that a plot with only 1 point will have all metrics automatically set to 0 or nan
    if np.array(h).size >= 2:

        # Calc Cover for height bins
        nbins = ((hmax - hmin) / step) + 1
        hbins = np.linspace(hmin, hmax, int(nbins))

        # IMPORTANT: Using groundthres, you may want to account for errors in relative accuracy
        # EX: IF the rel. accuracy of ground is about 0.06 m (6 cm) between flightlines,
        # the lowest height bin could be set to 0.06 (instead of 0) to account for this.
        # so any hit below 0.06 m counts as ground.
        # NOTE: If you want to use everything, just set groundthres to 0
        if groundthres > 0:
            # insert the groundthres into the array (right above 0)
            heightbins = np.insert(hbins, 1, groundthres)
            # heightbins[heightbins==0] = groundthres
        if groundthres < 0:
            # insert the groundthres into the array (right below 0)
            heightbins = np.insert(hbins, 0, groundthres)
            # heightbins[heightbins==0] = groundthres

        # sort points by height
        h = np.sort(h)

        # Group each point by height bin
        hgroups = np.digitize(h, bins=hbins)

        # Count the number of points in each bin
        # Note: minlength allows it to go to the full length of hbins, rather than stop at the max height of the points
        hcounts = np.bincount(hgroups, minlength=len(hbins))

        # Normalize the counts
        hcounts_norm = hcounts/np.sum(hcounts)

        # Smooth point height distribution, and find peaks and troughs\
        # Following the below article on stack exchange
        # https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

        # smooth with gaussian filter
        smooth = gaussian_filter1d(hcounts_norm, smoothsigma)

        # compute first derivative
        smooth_d1 = np.gradient(smooth)
        
        # Interpolate heights and derivative to 1 cm increments so that you have more precise inflection points
        hbins_interp= np.arange(0, np.max(hbins), 0.01)
        smooth_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth)
        smooth_d1_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth_d1)

        # find inflection points
        # when it switches from positive to negative (a trough)
        idx_troughs = np.where(np.diff(np.sign(smooth_d1_interp))>0)[0]
        # when it switches from negative to positive (a peak)
        idx_peaks = np.where(np.diff(np.sign(smooth_d1_interp))<0)[0]

        # output height values of peaks and troughs
        troughs = hbins_interp[idx_troughs]
        peaks = hbins_interp[idx_peaks]

        # Number of layers as number of peaks
        nlayers = len(peaks)
        
        if nlayers > 1:
            
            # height location of peak with largest frequency of points
            maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            
            # Get the gap size as the maximum distance between consecutive peaks
            gapsize = np.max(np.diff(peaks))
            # gapsize = gapsize.round(3)
            
            # MEAN and STD peak HEIGHT
            # made up by peter
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nanstd(peaks)
            cvpeakh = np.nanmean(peaks)/np.nanstd(peaks)
            
            # Round
            # stdpeakh = stdpeakh.round(3)
            # cvpeakh = cvpeakh.round(3)
            
            # Vertical Distribution Ratio (Goetz 2007) 
            # - experimental, computed from peaks instead of norm point distribution
            VDRpeak = (np.max(peaks) - np.nanmedian(peaks)) / np.max(peaks)
            # VDRpeak = np.round(VDRpeak, 3)
            
            
        else:
            
            # Else: set to nan, so that only pixels that
            # have multiple layers in them are sampled
            # try:
            #     maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            #     meanpeakh = np.nanmean(peaks)
            # except:
            #     maxpeakh = np.nan
            #     meanpeakh = np.nan
            
            # Edited 10/31/22 
            # If there's only 1 layer, 
            # set the peaks to be the only peak height
            if peaks.size:
                maxpeakh = peaks[0]
                meanpeakh = peaks[0]
            # Except if there is no peak to record
            else:
                maxpeakh = 0
                meanpeakh = 0
                
            # and variation metrics to 0
            stdpeakh = 0
            gapsize = 0
            cvpeakh = 0
            VDRpeak = 0

        # computing the PtoH ratio (Davies 2020, Asner 2014)
        perc99 = np.nanpercentile(h, 99, method='median_unbiased')
        ptoh = maxpeakh/perc99
        # ptoh = ptoh.round(3)

        # Complexity Score (Davies 2020)
        # proportion of bins with points in them vs without
        # could be insensitive to striping
        cscore = np.sum(hcounts>0)/len(hcounts)
        # cscore = cscore.round(3)

        # Vertical Distribution Ratio (Goetz 2007)
        VDR = (np.max(h) - np.median(h)) / np.max(h)
        # VDR = np.round(VDR, 3)

        # Foliage Height Diversity 
        # (Bergen 2009 and many others using PAI profile - this is using normalized point counts)
        if np.sum(hcounts_norm>0) > 0:
            FHD = -1*np.sum(hcounts_norm*np.log(hcounts_norm, where=hcounts_norm>0)) 
            # FHD = FHD.round(3)
        else:
            FHD = 0

        ### SAVE Outputs
        complex_dict = {'nlayers':nlayers,
                        'gapsize':gapsize,
                        'maxpeakh':maxpeakh,
                        'ptoh':ptoh,
                        'cscore':cscore,
                        'FHD':FHD,
                        'VDR':VDR,
                        'VDRpeak':VDRpeak,
                        'meanpeakh':meanpeakh,
                        'stdpeakh':stdpeakh,
                        'cvpeakh':cvpeakh
                       }

        if plot:

            fig, ax = plt.subplots()
            ax.plot(hcounts_norm, hbins, label='Raw', lw=2, alpha=0.6)
            ax.plot(smooth, hbins, label='Smoothed', lw=2)
            # for infl in troughs:
            #     tline = ax.axhline(y=infl, color='b', label='Trough', alpha=0.6)
            for infl in peaks:
                pline = ax.axhline(y=infl, color='c', label='Peak', alpha=0.6)
            # ax.legend(handles=[tline, pline], loc='best')
            ax.legend(loc='best')
            ax.set_xlabel('Normalized Frequency of Points')
            ax.set_ylabel('Height [m]')
            ax.set_xlim(-0.01, np.max(hcounts_norm) + 0.03)
            ax.set_ylim(0, np.max(hbins) + 0.3)

            # don't return peaks, troughs
            return complex_dict, fig, ax

        else:

            return complex_dict
    
    # Else, if there were no heights in the array
    # just return an array of 0s
    else:
        
        complex_dict = {'nlayers':0,
                        'gapsize':0,
                        'maxpeakh':0,
                        'ptoh':0,
                        'cscore':0,
                        'FHD':0,
                        'VDR':0,
                        'VDRpeak':0,
                        'meanpeakh':0,
                        'stdpeakh':0,
                        'cvpeakh':0
                       }
        
        return complex_dict


# New Complexity Function
# PB 10/31/22
# NOTE: If performing over a set of pixels, need to wrap the below in a loop
def canopyLayerMetrics(h, hmax, step, groundthres, hmin=0, plot=False, smoothsigma=2):
    # Where h is an array of point height values
    # hbins is a list/array of height bin edges
    # smoothsigma = positive float - gives a smoothing parameter for gaussian smoothing in layer calculation (in meters)
    # plot = True/False - whether to plot the results
    
    # If there are any heights in the array
    # edited 10/26 - have to have at least 2 points above the groundthreshold to do complexity stats
    # This means that a plot with only 1 point will have all metrics automatically set to 0 or nan
    if np.array(h).size >= 2:

        # Calc Cover for height bins
        nbins = ((hmax - hmin) / step) + 1
        hbins = np.linspace(hmin, hmax, int(nbins))

        # IMPORTANT: Using groundthres, you may want to account for errors in relative accuracy
        # EX: IF the rel. accuracy of ground is about 0.06 m (6 cm) between flightlines,
        # the lowest height bin could be set to 0.06 (instead of 0) to account for this.
        # so any hit below 0.06 m counts as ground.
        # NOTE: If you want to use everything, just set groundthres to 0
        if groundthres > 0:
            # insert the groundthres into the array (right above 0)
            heightbins = np.insert(hbins, 1, groundthres)
            # heightbins[heightbins==0] = groundthres
        if groundthres < 0:
            # insert the groundthres into the array (right below 0)
            heightbins = np.insert(hbins, 0, groundthres)
            # heightbins[heightbins==0] = groundthres

        # sort points by height
        h = np.sort(h)

        # Group each point by height bin
        hgroups = np.digitize(h, bins=hbins)

        # Count the number of points in each bin
        # Note: minlength allows it to go to the full length of hbins, rather than stop at the max height of the points
        hcounts = np.bincount(hgroups, minlength=len(hbins))

        # Normalize the counts
        hcounts_norm = hcounts/np.sum(hcounts)

        # Smooth point height distribution, and find peaks and troughs\
        # Following the below article on stack exchange
        # https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

        # smooth with gaussian filter
        smooth = gaussian_filter1d(hcounts_norm, smoothsigma)

        # compute first derivative
        smooth_d1 = np.gradient(smooth)
        
        # Interpolate heights and derivative to 1 cm increments so that you have more precise inflection points
        hbins_interp= np.arange(0, np.max(hbins), 0.01)
        smooth_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth)
        smooth_d1_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth_d1)

        # find inflection points
        # when it switches from positive to negative (a trough)
        idx_troughs = np.where(np.diff(np.sign(smooth_d1_interp))>0)[0]
        # when it switches from negative to positive (a peak)
        idx_peaks = np.where(np.diff(np.sign(smooth_d1_interp))<0)[0]

        # output height values of peaks and troughs
        troughs = hbins_interp[idx_troughs]
        peaks = hbins_interp[idx_peaks]

        # Number of layers as number of peaks
        nlayers = len(peaks)
        
        if nlayers > 1:
            
            # height location of peak with largest frequency of points
            maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            
            # Get the gap size as the maximum distance between consecutive peaks
            gapsize = np.max(np.diff(peaks))
            # gapsize = gapsize.round(3)
            
            # MEAN and STD peak HEIGHT
            # made up by peter
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nanstd(peaks)
            cvpeakh = np.nanmean(peaks)/np.nanstd(peaks)
            
            # Round
            # stdpeakh = stdpeakh.round(3)
            # cvpeakh = cvpeakh.round(3)
            
            # Vertical Distribution Ratio (Goetz 2007) 
            # - experimental, computed from peaks instead of norm point distribution
            VDRpeak = (np.max(peaks) - np.nanmedian(peaks)) / np.max(peaks)
            # VDRpeak = np.round(VDRpeak, 3)
            
            
        else:
            
            # Else: set to nan, so that only pixels that
            # have multiple layers in them are sampled
            # try:
            #     maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]
            #     meanpeakh = np.nanmean(peaks)
            # except:
            #     maxpeakh = np.nan
            #     meanpeakh = np.nan
            
            # Edited 10/31/22 
            # If there's only 1 layer, 
            # set the peaks to be the only peak height
            if peaks.size:
                maxpeakh = peaks[0]
                meanpeakh = peaks[0]
            # Except if there is no peak to record
            else:
                maxpeakh = 0
                meanpeakh = 0
                
            # and variation metrics to 0
            stdpeakh = 0
            gapsize = 0
            cvpeakh = 0
            VDRpeak = 0

        # computing the PtoH ratio (Davies 2020, Asner 2014)
        perc99 = np.nanpercentile(h, 99, method='median_unbiased')
        ptoh = maxpeakh/perc99
        # ptoh = ptoh.round(3)

        # Complexity Score (Davies 2020)
        # proportion of bins with points in them vs without
        # could be insensitive to striping
        cscore = np.sum(hcounts>0)/len(hcounts)
        # cscore = cscore.round(3)

        # Vertical Distribution Ratio (Goetz 2007)
        VDR = (np.max(h) - np.median(h)) / np.max(h)
        # VDR = np.round(VDR, 3)

        # Foliage Height Diversity 
        # (Bergen 2009 and many others using PAI profile - this is using normalized point counts)
        if np.sum(hcounts_norm>0) > 0:
            FHD = -1*np.sum(hcounts_norm*np.log(hcounts_norm, where=hcounts_norm>0)) 
            # FHD = FHD.round(3)
        else:
            FHD = 0

        ### SAVE Outputs
        complex_dict = {'nlayers':nlayers,
                        'gapsize':gapsize,
                        'maxpeakh':maxpeakh,
                        'ptoh':ptoh,
                        'cscore':cscore,
                        'FHD':FHD,
                        'VDR':VDR,
                        'VDRpeak':VDRpeak,
                        'meanpeakh':meanpeakh,
                        'stdpeakh':stdpeakh,
                        'cvpeakh':cvpeakh
                       }

        if plot:

            fig, ax = plt.subplots()
            ax.plot(hcounts_norm, hbins, label='Raw', lw=2, alpha=0.6)
            ax.plot(smooth, hbins, label='Smoothed', lw=2)
            # for infl in troughs:
            #     tline = ax.axhline(y=infl, color='b', label='Trough', alpha=0.6)
            for infl in peaks:
                pline = ax.axhline(y=infl, color='c', label='Peak', alpha=0.6)
            # ax.legend(handles=[tline, pline], loc='best')
            ax.legend(loc='best')
            ax.set_xlabel('Normalized Frequency of Points')
            ax.set_ylabel('Height [m]')
            ax.set_xlim(-0.01, np.max(hcounts_norm) + 0.03)
            ax.set_ylim(0, np.max(hbins) + 0.3)

            # don't return peaks, troughs
            return complex_dict, fig, ax

        else:

            return complex_dict
    
    # Else, if there were no heights in the array
    # just return an array of 0s
    else:
        
        complex_dict = {'nlayers':0,
                        'gapsize':0,
                        'maxpeakh':0,
                        'ptoh':0,
                        'cscore':0,
                        'FHD':0,
                        'VDR':0,
                        'VDRpeak':0,
                        'meanpeakh':0,
                        'stdpeakh':0,
                        'cvpeakh':0
                       }
        
        return complex_dict
    
    
# # Function for calculating percentile heights 
# Outdated verision in LabLidar_Functions.py- 10/31/22 commented out and replaced 
def calcPercentileHeights(points, groundthres=0, returnHeights=True, heightcol='HeightAboveGround'):
    
    # Calculate Percentile Metrics of Height
    perc_dict= {0:[],
                25:[],
                50:[],
                75:[],
                98:[],
                100:[],
                'mean':[],
                'std':[]}

    # Get Heights for given cell
    heights = points[heightcol]

    # If there are any heights left
    if heights.size > 0:

        perc_dict[0].append(np.quantile(heights, [0]).flat[0])
        perc_dict[25].append(np.quantile(heights, [0.25]).flat[0])
        perc_dict[50].append(np.quantile(heights, [0.5]).flat[0])
        perc_dict[75].append(np.quantile(heights, [0.75]).flat[0])
        perc_dict[98].append(np.quantile(heights, [0.98]).flat[0])
        perc_dict[100].append(np.quantile(heights, [1.0]).flat[0])
        perc_dict['mean'].append(np.mean(heights).flat[0])
        perc_dict['std'].append(np.std(heights).flat[0])

    # else, height stats are 0
    else:
        perc_dict[0].append(0)
        perc_dict[25].append(0)
        perc_dict[50].append(0)
        perc_dict[75].append(0)
        perc_dict[98].append(0)
        perc_dict[100].append(0)
        perc_dict['mean'].append(0)
        perc_dict['std'].append(0)

    if returnHeights:
        return perc_dict, heights

    else:
        return perc_dict



# Wrapper function for using parallel processing and calccover function 
# Notice that is calls PlotCloud as the first argument (an input specific to this script)
# It has to be this way in order to use concurrent futures parallel processing below.
def calccover_parallel(index, groundthres=groundthreshold):

    # make a True/False array 
    # for all points within the current grid cell
    idx_bool = PlotCloud.grid_dict['idx_points'] == PlotCloud.grid_dict['idx_cells'][index]
    
    try:
        
        # Calculate metrics
        cover = calccover(points=PlotCloud.las.points[idx_bool],
                          step=PlotCloud.vsize,
                          groundthres=groundthres,
                          heightcol=PlotCloud.heightcol,
                          hmax=maxh)

    except Exception as e:

        print(f"{e.__class__} for {lc.lasf}: \n")
        print(f"\t{e}\n")

    # Calculate Percentile Metrics of Height
    perc_dict= {0:[],
                25:[],
                50:[],
                75:[],
                98:[],
                100:[],
                'mean':[],
                'std':[]}
    
    # Get Heights for given cell
    heights = PlotCloud.las.points[PlotCloud.heightcol][idx_bool]
    
    # Make sure heights only includes points > ground threshold
    heights = heights[heights >= groundthres]
    
    # If there are any heights left
    if heights.size > 0:
        
        perc_dict[0].append(np.quantile(heights, [0]).flat[0])
        perc_dict[25].append(np.quantile(heights, [0.25]).flat[0])
        perc_dict[50].append(np.quantile(heights, [0.5]).flat[0])
        perc_dict[75].append(np.quantile(heights, [0.75]).flat[0])
        perc_dict[98].append(np.quantile(heights, [0.98]).flat[0])
        perc_dict[100].append(np.quantile(heights, [1.0]).flat[0])
        perc_dict['mean'].append(np.mean(heights).flat[0])
        perc_dict['std'].append(np.std(heights).flat[0])
    
    # else, height stats are 0
    else:
        perc_dict[0].append(0)
        perc_dict[25].append(0)
        perc_dict[50].append(0)
        perc_dict[75].append(0)
        perc_dict[98].append(0)
        perc_dict[100].append(0)
        perc_dict['mean'].append(0)
        perc_dict['std'].append(0)

    # Get Intensity for the given cell
    # intensity = PlotCloud.las.points['Intensity'][idx_bool]
    
#     # Fill dictionary for intensity metrics
#     int_dict= {0:[],
#                25:[],
#                50:[],
#                75:[],
#                98:[],
#                100:[],
#                'mean':[],
#                'std':[]}
    
#     int_dict[0].append(np.quantile(intensity, [0]).flat[0])
#     int_dict[25].append(np.quantile(intensity, [0.25]).flat[0])
#     int_dict[50].append(np.quantile(intensity, [0.5]).flat[0])
#     int_dict[75].append(np.quantile(intensity, [0.75]).flat[0])
#     int_dict[98].append(np.quantile(intensity, [0.98]).flat[0])
#     int_dict[100].append(np.quantile(intensity, [1.0]).flat[0])
#     int_dict['mean'].append(np.mean(intensity).flat[0])
#     int_dict['std'].append(np.std(intensity).flat[0])
     
    if calculatecover:
        # Return cover dict, percentile dict, and height list (for quick recalculation of anything later)
        return cover, perc_dict, heights
    else:
        return perc_dict, heights
    
    
    
# NOTE: If performing over a set of pixels, need to wrap the below in a loop
def canopyLayerMetrics(h, hmax, step, groundthres, hmin=0, plot=False, smoothsigma=2):
    # Where h is an array of point height values
    # hbins is a list/array of height bin edges
    # smoothsigma = positive float - gives a smoothing parameter for gaussian smoothing in layer calculation (in meters)
    # plot = True/False - whether to plot the results
    
    # If there are any heights in the array
    # edited 10/26 - have to have at least 3 points in array to do complexity stats
    # Edit 10/25/22 to deal with an error
    # if there's only 1 point in the array, it throws a 
    # "fp and xp are not of the same length." value error
    # I think b/c you can't interpolate with only 1 point
    # This means that a plot with only 1 point will have all metrics automatically set to 0 or nan
    if np.array(h).size >= 3:

        # Calc Cover for height bins
        nbins = ((hmax - hmin) / step) + 1
        hbins = np.linspace(hmin, hmax, int(nbins))

        # IMPORTANT: Using groundthres, you may want to account for errors in relative accuracy
        # EX: IF the rel. accuracy of ground is about 0.06 m (6 cm) between flightlines,
        # the lowest height bin could be set to 0.06 (instead of 0) to account for this.
        # so any hit below 0.06 m counts as ground.
        # NOTE: If you want to use everything, just set groundthres to 0
        if groundthres > 0:
            # insert the groundthres into the array (right above 0)
            heightbins = np.insert(hbins, 1, groundthres)
            # heightbins[heightbins==0] = groundthres
        if groundthres < 0:
            # insert the groundthres into the array (right below 0)
            heightbins = np.insert(hbins, 0, groundthres)
            # heightbins[heightbins==0] = groundthres

        # sort points by height
        h = np.sort(h)

        # Group each point by height bin
        hgroups = np.digitize(h, bins=hbins)

        # Count the number of points in each bin
        # Note: minlength allows it to go to the full length of hbins, rather than stop at the max height of the points
        hcounts = np.bincount(hgroups, minlength=len(hbins))

        # Normalize the counts
        hcounts_norm = hcounts/np.sum(hcounts)

        # Smooth point height distribution, and find peaks and troughs\
        # Following the below article on stack exchange
        # https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

        # smooth with gaussian filter
        smooth = gaussian_filter1d(hcounts_norm, smoothsigma)

        # compute first derivative
        smooth_d1 = np.gradient(smooth)
        
        # Interpolate heights and derivative to 1 cm increments so that you have more precise inflection points
        hbins_interp= np.arange(0, np.max(hbins), 0.01)
        smooth_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth)
        smooth_d1_interp = np.interp(x=hbins_interp, xp=hbins, fp=smooth_d1)

        # find inflection points
        # when it switches from positive to negative (a trough)
        idx_troughs = np.where(np.diff(np.sign(smooth_d1_interp))>0)[0]
        # when it switches from negative to positive (a peak)
        idx_peaks = np.where(np.diff(np.sign(smooth_d1_interp))<0)[0]

        # output height values of peaks and troughs
        troughs = hbins_interp[idx_troughs]
        peaks = hbins_interp[idx_peaks]
        
        # height location of peak with largest frequency of points
        maxpeakh = peaks[np.argmax(smooth_interp[idx_peaks])]

        # Number of layers as number of peaks
        nlayers = len(peaks)
        
        if nlayers > 1:
            
            # Get the gap size as the maximum distance between consecutive peaks
            gapsize = np.max(np.diff(peaks))
            # gapsize = gapsize.round(3)
            
            # MEAN and STD peak HEIGHT
            # made up by peter
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nanstd(peaks)
            cvpeakh = np.nanmean(peaks)/np.nanstd(peaks)
            
            # Round
            # stdpeakh = stdpeakh.round(3)
            # cvpeakh = cvpeakh.round(3)
            
            # Vertical Distribution Ratio (Goetz 2007) 
            # - experimental, computed from peaks instead of norm point distribution
            VDRpeak = (np.max(peaks) - np.nanmedian(peaks)) / np.max(peaks)
            # VDRpeak = np.round(VDRpeak, 3)
            
            
        else:
            
            # Else: set to nan, so that only pixels that
            # have multiple layers in them are sampled
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nan
            gapsize = np.nan
            cvpeakh = np.nan
            VDRpeak = np.nan

        # computing the PtoH ratio (Davies 2020, Asner 2014)
        perc99 = np.nanpercentile(h, 99, method='median_unbiased')
        ptoh = maxpeakh/perc99
        # ptoh = ptoh.round(3)

        # Complexity Score (Davies 2020)
        # proportion of bins with points in them vs without
        # could be insensitive to striping
        cscore = np.sum(hcounts>0)/len(hcounts)
        # cscore = cscore.round(3)

        # Vertical Distribution Ratio (Goetz 2007)
        VDR = (np.max(h) - np.median(h)) / np.max(h)
        # VDR = np.round(VDR, 3)

        # Foliage Height Diversity 
        # (Bergen 2009 and many others using PAI profile - this is using normalized point counts)
        if np.sum(hcounts_norm>0) > 0:
            FHD = -1*np.sum(hcounts_norm*np.log(hcounts_norm, where=hcounts_norm>0)) 
            # FHD = FHD.round(3)
        else:
            FHD = 0

        ### SAVE Outputs
        complex_dict = {'nlayers':nlayers,
                        'gapsize':gapsize,
                        'maxpeakh':maxpeakh,
                        'ptoh':ptoh,
                        'cscore':cscore,
                        'FHD':FHD,
                        'VDR':VDR,
                        'VDRpeak':VDRpeak,
                        'meanpeakh':meanpeakh,
                        'stdpeakh':stdpeakh,
                        'cvpeakh':cvpeakh
                       }

        if plot:

            fig, ax = plt.subplots()
            ax.plot(hcounts_norm, hbins, label='Raw', lw=2, alpha=0.6)
            ax.plot(smooth, hbins, label='Smoothed', lw=2)
            # for infl in troughs:
            #     tline = ax.axhline(y=infl, color='b', label='Trough', alpha=0.6)
            for infl in peaks:
                pline = ax.axhline(y=infl, color='c', label='Peak', alpha=0.6)
            # ax.legend(handles=[tline, pline], loc='best')
            ax.legend(loc='best')
            ax.set_xlabel('Normalized Frequency of Points')
            ax.set_ylabel('Height [m]')
            ax.set_xlim(-0.01, np.max(hcounts_norm) + 0.03)
            ax.set_ylim(0, np.max(hbins) + 0.3)

            # don't return peaks, troughs
            return complex_dict, fig, ax

        else:

            return complex_dict
    
    # Else, if there were no heights in the array
    # just return an array of nans
    else:
        
        complex_dict = {'nlayers':np.nan,
                        'gapsize':np.nan,
                        'maxpeakh':np.nan,
                        'ptoh':np.nan,
                        'cscore':np.nan,
                        'FHD':np.nan,
                        'VDR':np.nan,
                        'VDRpeak':np.nan,
                        'meanpeakh':np.nan,
                        'stdpeakh':np.nan,
                        'cvpeakh':np.nan
                       }
        
        return complex_dict