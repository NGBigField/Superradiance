classdef S < BaseSymbolicClass

    properties (SetAccess=immutable)
        script (1,1) string % {mustBeMember(script,["z","p","m"])}         
    end

    methods
        %%
        function obj = S(script)
            arguments
                script (1,1) string {mustBeMember(script,["z","+","-"])}
            end
            obj.script = standard_script(script);
        end       
        %%
        function res = unpack(obj)
            switch obj.script
                case "z"
                    res = obj.global_symbols.Sz;
                case "+"
                    res = obj.global_symbols.Sp;
                case "-"
                    res = obj.global_symbols.Sm;
                otherwise
                    error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
        end
        %%
        function res = multiply(obj, other)
            if isa(other, 'S')
                res = S.multiply_s_by_s(obj, other);
            elseif isa(other, 'sym')
                res = S.multiply_s_by_coef(obj, other);
            else
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
        end
        %%
        function res = simmilar(obj, other)
            if isa(other, 'S') && other.is( obj.script )
                res = true;
            else
                res = false;
            end
        end
        %%
        function tf = is(obj, target_script)
            arguments
                obj
                target_script (1,1) string {mustBeMember(target_script,["z","+","-"])}
            end
            tf = standard_script(target_script)==obj.script;
        end
        %%
    end % methods

    methods (Static)
                %%
        function res = multiply_s_by_s(s1, s2)
            arguments
                s1 (1,1) S
                s2 (1,1) S
            end
            res = SS(s1,s2);
        end
        %%
        function res = multiply_s_by_coef(s, c)
            arguments
                s (1,1) S
                c (1,1) sym
            end
            s.coef = s.coef * c;
            res = s;
        end
    end
end




function out = standard_script(script)
    switch script
        case {"Z","z"}
            out = "z";
        case {"+","P", "p"}
            out = "+";
        case {"-","M", "m"}
            out = "-";
        otherwise
            error("SymbolicClass:UnsupportedCase","Not a legit case");    
    end
end