function c = mycolormap(pivotal_colors)
% Function creating the colormap sweeping linearly from color c1 to color
% c2, color c3 etc... The various colors are contained in the rows of
% pivotal_colors
% pivotal_colors can also be a string of letters, each one corresponding to
% a color that will appear in sequence in the colorbar. The letters can be
% 'w'     White
% 'k'     Black
% 'b'     Blue
% 'g'     Green
% 'y'     Yellow
% 'p'     Purple
% 'r'     Red
% 'n'     brown


    if ischar(pivotal_colors)
        %If pivotal_colors contains a string specifying the colormap
        color_string = pivotal_colors;
        pivotal_colors = []; %Reinitialize the pivotal colors, that now become a true array of numbers
        
        for ncol = 1:length(color_string)
            color = color_string(ncol);
            switch color
                case 'w'
                    pivotal_colors = [pivotal_colors; 1,1,1];
                case 'k'
                    pivotal_colors = [pivotal_colors; 0,0,0];
                case 'b'
                    pivotal_colors = [pivotal_colors; 0,0,1];
                case 'g'
                    pivotal_colors = [pivotal_colors; 0.18, 0.66, 0.52];
                case 'y'
                    pivotal_colors = [pivotal_colors; 0.9,0.89,0.35];
                case 'p'
                    pivotal_colors = [pivotal_colors; 0.19,0.01,0.3];
                case 'r'
                    pivotal_colors = [pivotal_colors; 1,0,0];
                case 'n'
                    pivotal_colors = [pivotal_colors; 0.49,0.24,0.04];
                otherwise
                    display('ERROR: unvalid string for colormap specification')

            end
        end
    else
        %pivotal_colors should be a matrix containing the colors of the colorbar
        if size(pivotal_colors, 2) ~= 3
            display('ERROR: the input array containing the pivotal colors to create the colormap is not valid!!')
        end
    end
    
    Ncolors = size(pivotal_colors, 1);
    
    c = []; %Colormap initialization
    
    for nc = 1:Ncolors-1
        c1 = pivotal_colors(nc, :);
        c2 = pivotal_colors(nc+1, :);
        
        %Sweep over the colorbar for the various channels R, G, B
        Rsweep = linspace(c1(1), c2(1))';
        Gsweep = linspace(c1(2), c2(2))';
        Bsweep = linspace(c1(3), c2(3))';
        
        c = [c; [Rsweep, Gsweep, Bsweep]]; %Append the sweep among the two considered colors
    end
end