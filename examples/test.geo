
Point(newp) = {0, 0, 0, 1.0};
Point(newp) = {5, 0, 0, 1.0};
Point(newp) = {5, 5, 0, 1.0};
Point(newp) = {3, 3, 0, 1.0};
Point(newp) = {0, 5, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 1};

Curve Loop(1) = {3, 4, 5, 1, 2};
Plane Surface(1) = {1};

Physical Surface("steel", 6) = {1};
Physical Curve("dirichlet", 7) = {4, 5, 1, 2, 3};
