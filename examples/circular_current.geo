SetFactory("OpenCASCADE");

R = 1.0;
Rout = 4.0;
lc_inner = 0.25;
lc_outer = 0.55;

Point(1) = {0, 0, 0, lc_inner};

Point(2) = {R, 0, 0, lc_inner};
Point(3) = {0, R, 0, lc_inner};
Point(4) = {-R, 0, 0, lc_inner};
Point(5) = {0, -R, 0, lc_inner};

Point(6) = {Rout, 0, 0, lc_outer};
Point(7) = {0, Rout, 0, lc_outer};
Point(8) = {-Rout, 0, 0, lc_outer};
Point(9) = {0, -Rout, 0, lc_outer};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Circle(5) = {6, 1, 7};
Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};
Circle(8) = {9, 1, 6};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2, 1};

Physical Surface("coil", 1) = {1};
Physical Surface("air", 2) = {2};
Physical Curve("gap", 3) = {1, 2, 3, 4};
Physical Curve("dirichlet", 4) = {5, 6, 7, 8};
