classdef Config
    %Config Global configuration
    %   Can't be changed throughout the execution.
    properties (Constant)
        num_elements (1,1) int32   = 4
        simplify     (1,1) logical = true
        simplified_coefficients (1,1) logical = true
    end    
end

