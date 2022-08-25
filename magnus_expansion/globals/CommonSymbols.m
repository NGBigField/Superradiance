classdef CommonSymbols
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        E            (:,1)
        t            (:,1)
        d            (1,1)
        w            (1,1)
        Sp           (1,1)
        Sm           (1,1)
        Sz           (1,1)
        d_h          (1,1) % d over h_bar
    end

    methods
        function obj = CommonSymbols()
            num_elements (1,1) = Config().num_elements;
            % Define symbols:
            syms('E',[num_elements, 1]);
            syms('t',[num_elements, 1]);
            syms('w');
            syms('Sp');
            syms('Sm');
            syms('Sz');
            syms('d_h');  % d over h_bar
            % Assign symbols:
            obj.E = E;
            obj.t = t;
            obj.w       = w;      
            obj.Sp      = Sp;     
            obj.Sm      = Sm;     
            obj.Sz      = Sz;                 
            obj.d_h     = d_h;
        end
    end


    
end

