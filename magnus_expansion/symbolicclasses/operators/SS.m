classdef SS < BaseSymbolicClass
    properties
        S1 (1,1)  
        S2 (1,1)  
    end
    
    methods
        %%
        function obj = SS(S1, S2)
            arguments
                S1 (1,1) S
                S2 (1,1) S
            end
            obj.S1 = S1;
            obj.S2 = S2;
        end       
        %%
        function res = unpack(obj)
            res = ~obj.S1 * ~ obj.S2 ;     
        end
        %%
        function res = multiply(obj, other)
            error("SymbolicClass:Multiplication:NotSupported","Multiplication of S1S2 with another operation is not supported");    
        end
        %%
        function res = simmilar(ss1, ss2)            
            % Assert both are of S1S2 type:
            if ~isa(ss2, 'SS')
                error("SymbolicClass:NotImplemented", "This function is not implemented");
            end
            %% Check all versions:
            ss1.get(1)
        end
        %%
        function res = get(obj, i)
            % Return as requested:
            switch i
                case 1
                    res = obj.S1;
                case 2                    
                    res = obj.S2;
                otherwise
                    error("SymbolicClass:UnsupportedCase","Not a legit case");
            end
        end
        %% 
    end
end

%{
            % Both are S type:
            total_coef = s1.coef*s2.coef;            
            if s1.script == s2.script
                res = Product(s1, s2);

            elseif s1.script == "z"  % Our cannonical form is S_+\- before S_z
                res2 = MulOpsSymbolicClass();
                res = SumOpsSymbolicClass();
                if s2.script == "+"
                    relative_coef = +1;
                elseif s2.script == "-"
                    relative_coef = -1;
                end
                res1 = S( s2.script );
                res1.coef = relative_coef;
                res2.sub_operators = { S( s2.script ), S("z") };
                res.sub_operators = {res1, res2};               
                
            elseif ismember(s1.script, ["+", "-"] ) % keep same order
                    res = Product( S(s1.script), S(s2.script) );                    
            else 
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
            res.coef = total_coef;  
%}

