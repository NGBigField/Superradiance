function [out] = f(a, options)
    arguments
        a         (1,1) int32
        options.d  (1,1) Direction = Direction.up
    end
    d = options.d;
    
    switch d
        case Direction.up    , out = a + 1;
        case Direction.down  , out = a + 2;
        case Direction.left  , out = a + 3;
        case Direction.right , out = a + 4;
        otherwise
            error("option d="+string(d)+" of type '"+string(class(d))+"' is not valid.")
    end
end