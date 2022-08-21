classdef S < BaseSymbolicClass
    properties
        script (1,1) string % {mustBeMember(script,["z","p","m"])} 
    end
    
    methods
        function obj = S(script)
            arguments
                script (1,1) string {mustBeMember(script,["z","Z","+","-","p","m","P","M"])}
            end
            obj.script = standard_script(script);
        end       

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

        function res = multiply(obj, other)
            if ~isa(other, 'S')
                error("SymbolicClass:NotImplemented","Not implemented");    
            end
            % Both are S type:
            total_coef = obj.coef*other.coef;            
            if obj.script == other.script
                res = Product(obj, other);

            elseif obj.script == "z"  % Our cannonical form is S_+\- before S_z
                res2 = MulOpsSymbolicClass();
                res = SumOpsSymbolicClass();
                if other.script == "+"
                    relative_coef = +1;
                elseif other.script == "-"
                    relative_coef = -1;
                end
                res1 = S( other.script );
                res1.coef = relative_coef;
                res2.sub_operators = { S( other.script ), S("z") };
                res.sub_operators = {res1, res2};               
                
            elseif ismember(obj.script, ["+", "-"] ) % keep same order
                    res = Product( S(obj.script), S(other.script) );                    
            else 
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
            res.coef = total_coef;    

            

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
            error("Not supported");
    end
end