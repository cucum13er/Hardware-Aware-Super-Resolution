function TransM = get_affine_knownScale(scale, pts1, pts2)
syms a12 a13 a21 a23
A = [scale, a12, a13;
     a21, scale, a23;
       0,     0,   1;];
eqn = pts2 == A * pts1;

[AA,b] = equationsToMatrix(eqn,[a12,a13,a21,a23]);
X = lsqr(double(AA),double(b),1e-6,1e3);
A = subs(A,{a12, a13, a21, a23},{X(1), X(2), X(3), X(4)});
TransM = double(A);    
end