function sphere2png(exportFolder, tag, dpi)
%Function that exports a sphere plot in png

%INPUTS
%exportFolder   String with name of the folder in which you want the
%               export. Set to [] if you do not want any. If the folder
%               does not exist, one is created.
%tag            Tag to name the figures
%dpi            dots per inch - to regulate resolution (300 is good)


    if ispc; delimiter = '\'; else; delimiter = '/'; end;                       %Delimiters are /or \ depending on the operating system
    
    if length(exportFolder) > 0 & ~exist(exportFolder, 'dir')
       mkdir(exportFolder);                         %Create the export folder if it does not exist
    end
    
    ax = gca; %Axes to be copied
    
    xl = xlim; yl = ylim; cax = caxis; cmap = colormap; %Save things that have to be preserved
    
    %Export the surf
    dumbfig = figure('Visible','off');              %Open an invisible dumb figure
    new_ax = copyobj(ax,dumbfig);                   %Copy the axes of interest onto the dumbfigure
    xlim(xl); ylim(yl); caxis(cax); colormap(cmap); %Set things to match the original ones
    axis equal;
    new_ax.Position = [0 0 1 1];                    %Extend axes to the whole available figure
    new_ax.Visible = 'off';
    set(gcf,'Color',[1 1 1 0]);                     %Change background
    dumbfig.Position = [dumbfig.Position([1,2,4,4])];%Make figure squared
    filename = [pwd, delimiter, exportFolder, delimiter, tag, '_SPHERE.png'];
    print(dumbfig,filename,'-dpng',['-r', num2str(dpi)]);
    close(dumbfig);
end