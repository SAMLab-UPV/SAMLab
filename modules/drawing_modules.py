"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.

Commercial licenses are available. Contact: rmiralle@dcom.upv.es
"""

from librosa import stft as librosa_stft # For STFT theoretically faster than spectrogram
import numpy as np
from pandas import unique, isna # np unique does not preserv the order and sort!
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # To draw rectangle in miniature view of time signal

from .ui_constants import LABEL_FONT_SIZE,EVENT_TEXT_FONT_SIZE

def draw_tfr(self,y,fs,posx):
        
        #tstart = time.time() 
        posx_s=posx/fs  # slider pos in seconds
        duration_s=len(y)/fs

        if (self.lf_zoom_enabled.isChecked()):
            fmin=float(self.zoom_freq_low_limit)
            fmax=float(self.zoom_freq_up_limit)
            fft_npoints=2**(self.FFTpoints.currentIndex()+5)
            hop_length_val=max(1, int(fft_npoints * 0.05)) # 95% overlap
            if self.hann_w_enabled.isChecked():
                Pxx =librosa_stft(y.astype(float), n_fft=fft_npoints, hop_length=hop_length_val,window="hann", center=True)
            else:
                Pxx =librosa_stft(y.astype(float), n_fft=fft_npoints, hop_length=hop_length_val,window="hamming", center=True)

            freqs = np.arange(0, 1 + fft_npoints / 2) * fs / fft_npoints
            freq_slice = np.where((freqs >= fmin) & (freqs <= fmax))
            # keep only frequencies of interest
            freqs   = freqs[freq_slice]
            Pxx = Pxx[freq_slice,:][0]
            self.TFR_bitmap.set_extent([posx_s,posx_s+duration_s,fmin,fmax])
            self.ax0.set_ylim(fmin,fmax)
        else:
            Pxx =librosa_stft(y.astype(float), n_fft=256, hop_length=64, center=True)
            self.TFR_bitmap.set_extent([posx_s,posx_s+duration_s,0,fs/2])
            self.ax0.set_ylim(0,fs/2)

        # EFFICIENT Draw the Spectrogram View
        #bitmap=20*np.log10(np.abs(Pxx))
        bitmap=20*np.log10(np.maximum(np.abs(Pxx), np.finfo(float).eps)) # Avoid log of zero by using a small epsilon value
        if not self.manual_CL_chkbox_enabled.isChecked():
             self.TFR_bitmap.set_clim(np.min(bitmap),np.max(bitmap))
        #bitmap=bitmap.astype(np.uint8) # Is this faster?
        self.TFR_bitmap.set_data(bitmap)
        
        # EFFICIENT Draw the time view in figTime axis
        ttimes=np.linspace(0, duration_s, num=len(y))+posx_s
        self.ax2.set_ylim(np.min(y), np.max(y))
        self.ax2.set_xlim(posx_s, posx_s+duration_s)
        self.time_view_lines.set_data(ttimes,y)

        # DELETE ALL PREVIOUS EVENT MARKERS (Tag='EM')
        for artist in list(self.ax0.get_children()):
            if artist.get_gid() == "EM":
                artist.remove()
        
        if not self.events.empty:
            # FILTER EVENTS WITHIN DISPLAY RANGE
            evii = (self.events["end"] > posx) & (self.events["start"] < (posx + self.ventana - 1))
            events2 = self.events[evii].copy()

            # Filter by type
            #if self.eventstobeshown:
            events2 = events2[events2["type"].isin(self.eventstobeshown)]

            if not events2.empty:
                # Pre-allocate rectangle outlines
                Emark_x = []
                Emark_y = []

                # ------------------------------------------------------------
                # 3. LOOP EVENTS AND DRAW RECTANGLES, TEXT, CONTOURS
                # ------------------------------------------------------------
                for i, ev in events2.iterrows():
                    tiev = ev["start"] / fs
                    tfev = ev["end"]   / fs
                    fiev = ev["fmin"]
                    ffev = ev["fmax"]
                    # Rectangle points
                    Emark_x += [tiev, tfev, tfev, tiev, tiev, np.nan]
                    Emark_y += [fiev, fiev, ffev, ffev, fiev, np.nan]

                    # --------------------------------------------------------
                    # LABEL (MATLAB text(...,'Tag','EM'))
                    # --------------------------------------------------------
                    txt = self.ax0.text(
                        tfev, ffev,
                        ev["tag"],
                        fontsize=EVENT_TEXT_FONT_SIZE,
                        fontweight="bold",
                        color="blue",
                        rotation=45,
                        clip_on=True
                    )
                    txt.set_gid("EM")   # IMPORTANT

                    # --------------------------------------------------------
                    # EVENT CONTOUR (Tdata / Fdata) IF EXISTS
                    # --------------------------------------------------------
                    if not np.any(isna(ev["Tdata"])): # Proceed only if no NaNs at all
                
                        Tvals = np.array(ev["Tdata"])
                        Fvals = np.array(ev["Fdata"])

                        color = self.color_palette_WC[i % len(self.color_palette_WC)] # Cycle through colors
                        contour = self.ax0.plot(
                            Tvals, Fvals,
                            linestyle='-',
                            linewidth=1.5,
                            color=color
                        )[0]
                        contour.set_gid("EM")

                # ------------------------------------------------------------
                # 4. DRAW ALL RECTANGLES (MATLAB line(Emark_x,Emark_y,'Tag','EM'))
                # ------------------------------------------------------------
                rects = self.ax0.plot(
                    Emark_x, Emark_y,
                    linestyle="--",
                    linewidth=1.0,
                    color="blue"
                )[0]
                rects.set_gid("EM")

        #  Plot manual annotated events if within the visualization range
        if self.m_annotated_events and self.filer_manual_anotations_action.isChecked():
            # FILTER MANUAL ANNOTATED EVENTS WITHIN DISPLAY RANGE (Not Pandas. It is a list of dicts!)
            t0 = posx / fs
            t1 = (posx + self.ventana - 1) / fs
            events2 = [
                e for e in self.m_annotated_events
                if (e["tfin"] > t0) and (e["tini"] < t1)
            ]
            # Pre-allocate rectangle outlines
            Emark_x = []
            Emark_y = []
            # ------------------------------------------------------------
            # 3. LOOP EVENTS AND DRAW RECTANGLES, TEXT, CONTOURS
            # ------------------------------------------------------------
            for i, ev in enumerate(events2):
                tiev = ev["tini"]
                tfev = ev["tfin"]
                fiev = ev["freqmin"]
                ffev = ev["freqmax"]
                # Rectangle points
                Emark_x += [tiev, tfev, tfev, tiev, tiev, np.nan]
                Emark_y += [fiev, fiev, ffev, ffev, fiev, np.nan]

                # --------------------------------------------------------
                # LABEL (MATLAB text(...,'Tag','EM'))
                # --------------------------------------------------------
                txt = self.ax0.text(
                    tfev, ffev,
                    ev["m_event_type"],
                    fontsize=EVENT_TEXT_FONT_SIZE,
                    fontweight="bold",
                    color="black",
                    rotation=45,
                    clip_on=True
                )
                txt.set_gid("EM")   # IMPORTANT
            # ------------------------------------------------------------
            # 4. DRAW ALL RECTANGLES (MATLAB line(Emark_x,Emark_y,'Tag','EM'))
            # ------------------------------------------------------------
            rects = self.ax0.plot(
                Emark_x, Emark_y,
                linestyle="--",
                linewidth=1.0,
                color="black"
            )[0]
            rects.set_gid("EM")
        self.ax0.set_xlim(posx_s, posx_s+duration_s)
        self.ax0.figure.canvas.draw_idle()
        #print('Time:' ,(time.time()-tstart))


def big_data_graph(self,parameter,graph_yticks,TIME_STAMP_day,TIME_STAMP_min):
    graph_sep_line=2 # Vertical Graphic separation line (between days)
    default_pixx = 5
    pixy=15+graph_sep_line # Pixel size y (one file)

    graphsizex = 1440+default_pixx  # 24*60=1440 minutes/per day -> DO NOT CHANGE THIS!
    # I add "default_pixx" minutes in case the recording start at minute 59 to give space for the pixel in the graph
    graph = np.zeros((pixy * (int(np.max(TIME_STAMP_day)) + 1), graphsizex))
    
    n=len(parameter)
    for i in range(n):
        gx = int(TIME_STAMP_min[i])
        gy = int(TIME_STAMP_day[i])  # If the filename start in day "1" -> "TIME_STAMP_day[i]-1" if not "TIME_STAMP_day[i]"

        # Case 1: there is a next sample in the same day -> use that distance
        if i < n - 1 and TIME_STAMP_day[i]==TIME_STAMP_day[i + 1]:
            next_gx = int(TIME_STAMP_min[i + 1])
            pixx = next_gx - gx

        # Case 2: end of day -> use previous interval from same day
        else:
            prev_gx = int(TIME_STAMP_min[i - 1])
            pixx = gx - prev_gx

        graph[gy * pixy:gy * pixy + pixy - graph_sep_line, gx:gx + pixx].fill(parameter[i])
    graph_h, graph_w = graph.shape

    self.deployment_nav_bitmap.set_data(graph)
    self.deployment_nav_bitmap.set_clim(np.min(graph),np.max(graph))
    #self.deployment_nav_bitmap.set_clim(80,119)
    ## Make data coordinates = pixels 
    self.ax4.set_xlim(0, graph_w)
    self.ax4.set_ylim(graph_h, 0)

    # Locate and assign colors to weekend days
    graph_yticks_colors = graph_yticks.astype(str)  # Graph Yticks with weekend marks
    for l in range(len(graph_yticks)):
         newdata=datetime.strptime(graph_yticks[l],'%d-%b-%Y')
         if newdata.weekday()==5 or newdata.weekday()==6:
              graph_yticks_colors[l] = "r"
         else:
              graph_yticks_colors[l] = "k"
    
    posyticks = unique(TIME_STAMP_day.ravel())
    self.ax4.set_yticks(posyticks * pixy + np.ceil(pixy / 2))
    self.ax4.set_yticklabels(graph_yticks)
    graph_xticks = np.array([f'{i:02d}' for i in range(25)])
    posxticks = np.round(np.linspace(0, graphsizex, 25)).astype(int)
    self.ax4.set_xticks(posxticks)
    self.ax4.set_xticklabels(graph_xticks)

    # Minor ticks -> small marks at row boundaries
    minor_ticks = posyticks * pixy
    self.ax4.set_yticks(minor_ticks, minor=True)

    self.ax4.tick_params(axis='both', which='major', length=0,labelsize=LABEL_FONT_SIZE)
    self.ax4.tick_params(axis='y', which='minor',direction='in',length=0,left=True,right=False) # Minor ticks length 0 to hide them but will be used for the grid
    # Draw dashed horizontal lines at minor tick positions
    self.ax4.grid(which='minor',axis='y',linestyle='--',linewidth=0.7,color='gray',alpha=0.2)
    
    for ytick, graph_yticks_colors in zip(self.ax4.get_yticklabels(), graph_yticks_colors):
        ytick.set_color(graph_yticks_colors)

    self.deployment_nav_bitmap.set_extent([0.5, graph_w + 0.5, 0.5, graph_h + 0.5]) 
    #self.ax4.yaxis.set_inverted(True)     # To reverse YDir
    self.figDeployInspect.canvas.draw_idle()

def draw_auxiliary_nav_graph(self):
    #
    #   draw_auxiliary_nav_graph
    #
    #   Draw auxilairy navigation graph as an X/Y graph for the SAMLab software
    #   showing the miniature of the time signal and the different events in
    #   time versus a physical meaningful feature (SPL, ICI, f0)
    #
    #
    #    # INPUTS (as class attributes):
    #    - ax3 (axis for the auxiliary nav graph)
    #    - mini_x (vector simplified representation of the time signal)
    #    - totlength (samples in the signal)
    #    - fs
    #    - select_SPL_graph (bool)
    #    - select_ICI_graph (bool)
    #    - select_f0_graph (bool)
    #    - events: pandas DataFrame with columns like 'start', 'SPL', 'ICI', 'f0', 'type', 'tag'
    #

    fontsize_labels = 10
    legendfontsize = 10
    scatter_event_marker_size = 10

    # ----------------------------
    #  1. DELETE OLD PLOTS
    # ----------------------------
    # for artist in self.ax3.get_children():
    #     if isinstance(artist, (plt.Line2D, plt.Text, plt.Patch, plt.Collection)):
    #         artist.remove()
    self.ax3.clear()
    
    # ----------------------------
    # 2. SELECT EVENT PROPERTY TO SHOW
    # ----------------------------
    if self.select_SPL_graph.isChecked():
        event_char_to_show = self.events['SPL'].values
        ylabtext = r"SPL (dB re 1 $\mu$Pa$^2$)"
    elif self.select_ICI_graph.isChecked():
        event_char_to_show = self.events['ICI'].values
        ylabtext = "IEI (sec.)"
    elif self.select_f0_graph.isChecked():
        event_char_to_show = self.events['f0'].values
        ylabtext = r"$f_0$ (Hz)"
    else:
        event_char_to_show = np.array([])
        ylabtext = ""

    mini_x = self.mini_x
    fs = self.fs
    
    # ----------------------------
    # 3. MINIATURE TIME SIGNAL
    # ----------------------------
    mini_t=np.linspace(0, self.siz/self.fs, num=self.mini_x.size)

    # Adapt the miniature y-range to that of the events
    if not self.events.empty:
        maxeSPL = np.nanmax(event_char_to_show) + 0.1
        mineSPL = np.nanmin(event_char_to_show) - 0.1
        scale_y = (maxeSPL - mineSPL) / 2
        midpoint = scale_y + mineSPL
    else:
        scale_y = 1
        midpoint = 0

    # DRAW THE TIME VIEW IN THE figMiniature axis (ax3)
    self.ax3.fill(np.concatenate((mini_t, mini_t[::-1])),np.concatenate((self.mini_x,-self.mini_x[::-1]))*scale_y+midpoint,alpha=0.4)
    self.ax3.margins(x=0)

    # ----------------------------
    # 4. AXES LABELS AND LIMITS
    # ----------------------------
    self.ax3.set_xlabel('Time (seconds)', fontsize=fontsize_labels) # removed 'fontname'='Times'
    self.ax3.set_ylabel(ylabtext, fontsize=fontsize_labels) # removed 'fontname'='Times'
    #self.ax3.set_xlim([0, totlength / fs])
    self.ax3.set_xlim([0, self.siz / fs])
    if len(mini_x) > 0:
        self.ax3.set_ylim([-np.max(mini_x), np.max(mini_x)])
    self.ax3.grid(True)

    # ----------------------------
    # 5. PLOT EVENTS
    # ----------------------------
    if not self.events.empty:
        legend_elements = []
        legend_handles = []
        detected_event_types = self.events['type'].unique()
        colorpalette = plt.cm.tab10(np.linspace(0, 1, len(detected_event_types)))

        for l, evtype in enumerate(detected_event_types):
            ev_idx = self.events.index[self.events['type'] == evtype].tolist()
            ev_values = event_char_to_show[ev_idx].astype(float)

            if not all(np.isnan(ev_values)):
                # Scatter
                hl=self.ax3.scatter(
                    self.events.loc[ev_idx, 'start'] / fs,
                    ev_values,
                    s=scatter_event_marker_size,
                    color=colorpalette[l],
                    zorder=3
                )

                # Median line
                vm = np.nanmedian(ev_values)
                hlm = self.ax3.hlines(vm,0,self.mini_x.size,
                    colors=colorpalette[l],
                    linestyles='--',
                    zorder=3,linewidth=1
                )

                legend_elements.append(self.events.loc[ev_idx[0], 'tag'])
                legend_handles.append(hl)

        # ----------------------------
        # 6. LEGEND
        # ----------------------------
        if legend_elements:
            hle = self.ax3.legend(legend_handles,legend_elements, loc='lower center', fontsize=legendfontsize, bbox_to_anchor=(0.5, -1.2))
            self.ax3.add_artist(hle)
        self.ax3.set_ylim(mineSPL, maxeSPL)

    if self.m_annotated_events:
        all_tini = [event["tini"] for event in self.m_annotated_events]
        if not self.events.empty:
            ev_values = np.ones(len(all_tini))*ev_values.mean() 
        else:
            ev_values = np.zeros(len(all_tini)) # If there are already events, put the manual ones at the mean of the existing ones. If not, just put them at 0.
        # Scatter
        hl=self.ax3.scatter(
            all_tini,
            ev_values,
            s=scatter_event_marker_size,
            color="black",
            zorder=3
            )

    [miny, maxy]=self.ax3.get_ylim()
    self.rect_miniature_view=self.ax3.add_patch( Rectangle((self.posx/self.fs, miny),
                    self.tanalisis, maxy-miny,
                    #fc='none',
                    edgecolor ='red',
                    linewidth = 1,
                    linestyle="dashdot",
                    facecolor=(1, 0, 0, 0.1),   # red with 10% opacity
                    #alpha=0.1,
                    zorder=20) )
    
    self.figMiniature.canvas.draw_idle()
    #self.ax3.figure.canvas.draw_idle()

