function out = folder2var(folder_path)
    arguments
        folder_path (1,1) string
    end

    % Constants:    
    s = @(c) string(c);
    fs = s(filesep);
    
    % init output;
    out = struct();

    % Go over all subfolders
    listing = dir(folder_path);
    for i = 1 : length(listing)
        item = listing(i);
        name = item.name;
        if ismember( s(name), ["." ".."] )
            continue 
        end
        fullpath = s(item.folder)+fs+s(name);
        % Read file:
        text = string( textread(fullpath,"%s"));
        % save file:
        className = strsplit(s(name),'.');
        className = className(1);
        out.(className) = text;
    end

end