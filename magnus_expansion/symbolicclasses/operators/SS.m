classdef SS < Product

    methods
        %%
        function obj = SS(s1, s2)
            arguments
                s1 (1,1) S
                s2 (1,1) S
            end
            obj@Product(s1, s2)
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
            assert(i<=2 || i>=1)
            res = obj.subs{i};
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

