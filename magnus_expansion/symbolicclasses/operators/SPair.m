classdef SPair < Product

% Supresss messsages:
%#ok<*OR2> 

    properties
        s1 (1,1) string 
        s2 (1,1) string 
    end

    methods
        %%
        function obj = SPair(s1, s2)
            arguments
                s1 (1,1) S
                s2 (1,1) S
            end
            obj@Product(s1, s2)
            obj.s1 = s1.script;
            obj.s2 = s2.script;
        end       
        %%
        function res = multiply(obj, other)
            if isa(other, 'sym') || isnumeric(other)
                obj.coef = obj.coef * other;
            else
                error("SymbolicClass:Multiplication:NotSupported","Multiplication of S1S2 with another operation is not supported");
            end
            res = obj;
        end
        %%
        function res = reduce(a,b)
            arguments 
                a (1,1) SPair
                b (1,1) SPair
            end
            % assert reducable
            [is_similar, similarity_order] = permutations_similar(a,b);
            assert(is_similar);
            % reduce:
            if similarity_order==PairOrder.Given
                new = a;
                new.coef = a.coef + b.coef;
                if new.coef == 0
                    res = ZeroOperator;
                else
                    res = new;
                end
            elseif similarity_order==PairOrder.Commuted
                [ b_commuted, residue ] = b.commute();
                reduced = a;
                reduced.coef = a.coef + b_commuted.coef;
                res = {reduced, residue};
            else
                error("SymbolicClass:Bug:UnexpectedCase", "Code should not have reached this point");
            end
        end
        %%
        function [commuted, rest] = commute(obj)       
            % Prepare s1 and s2:
            S1 = S(obj.s1);
            S2 = S(obj.s2);
            % Switch places:
            commuted = SPair( S2, S1 ) * obj.coef;              
            % Find rest:
            rest = [ S1, S2 ] * obj.coef;

        end
        %%
        function is_similar = simmilar(a, b)            
            % Assert both are of S1S2 type:
            if ~isa(b, 'SPair')
                is_similar = false;
                return 
            end
            %% Check all versions:
            [is_similar, ~] = permutations_similar(a, b);
        end
        %%
        function [is_similar, pair_order] = permutations_similar(a, b)
            arguments 
                a (1,1) SPair
                b (1,1) SPair
            end
            if ( a.s1 == b.s1 ) && ( a.s2 == b.s2 )
                is_similar = true;
                pair_order = PairOrder.Given;
            elseif ( a.s1 == b.s2 ) && ( a.s2 == b.s1 )
                is_similar = true;
                pair_order = PairOrder.Commuted;
            else
                is_similar = false;
                pair_order = PairOrder.None;
            end
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

