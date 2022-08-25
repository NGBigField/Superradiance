classdef (Abstract) Callable


    methods (Abstract)
        out = call(inputs);
    end

    methods

    % Subsref method, modified to make scalar objects callable
    function varargout = subsref(obj, s)
      if strcmp(s(1).type, '()') && (numel(obj) == 1)
          % "__call__" equivalent: raise stored value to power of input
          input = s(1).subs{1};
          out = obj.call(input);
          varargout = {out};
      else
          % Use built-in subscripted reference for vectors/matrices
          varargout = {builtin('subsref', obj, s)};
      end
    end

  end

end