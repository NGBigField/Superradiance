function display_progress_bar(n,N)
%Function that displays the progress bar.
%n is the current step and N is the total number of steps
    
    if n == 1
        %Initialization of the progress bar
        progress_bar = [];
        for nc = 1:50
            progress_bar = [progress_bar, '_'];
        end
        fprintf(progress_bar);
    else
        perc = floor(n/N*100); %Percentage of the progress
        if mod(perc,2) == 0
            %Build new progress bar
            progress_bar = [];
            for nc = 1:perc/2
                progress_bar = [progress_bar, '#'];
            end
            for nc = 1:50-perc/2
                progress_bar = [progress_bar, '_'];
            end
            
            %Clear previous progress bar
            for nc = 1:50
                fprintf('\b');
            end
            
            %Display updated progress bar
            fprintf(progress_bar);
        end 
    end

end