classdef CommonSymbols
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        E            (:,1)
        t            (:,1)
        d            (1,1)
        h_bar        (1,1)
        w            (1,1)
        Sp           (1,1)
        Sm           (1,1)
        Sz           (1,1)
    end

    methods
        function obj = CommonSymbols()
            num_elements (1,1) = Config().num_elements;
            % Define symbols:
            syms('E',[num_elements, 1]);
            syms('t',[num_elements, 1]);
            syms('d') ;
            syms('h_bar');
            syms('w');
            syms('Sp');
            syms('Sm');
            syms('Sz');
            % Assign symbols:
            obj.E = E;
            obj.t = t;
            obj.d       = d;  
            obj.h_bar   = h_bar;  
            obj.w       = w;      
            obj.Sp      = Sp;     
            obj.Sm      = Sm;     
            obj.Sz      = Sz;                 
        end
    end


    
end

