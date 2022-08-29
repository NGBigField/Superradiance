classdef (Abstract) Callable


    methods (Abstract)
        out = call(inputs);
    end

    methods
        % Subsref method, modified to make scalar objects callable
        function varargout = subsref(obj, s)
            if strcmp(s(1).type, '()') && (numel(obj) == 1)
                % like "__call__" in python:
                input = s(1).subs{1};
                out = obj.call(input);
                varargout = {out};
            else
                N = nargout;
                varargout = cell(1,N);
                [varargout{:}] = builtin('subsref', obj, s);
            end
        end %function

    end %methods

end %classdef