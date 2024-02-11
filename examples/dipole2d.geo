
Point(newp) = {-2, 0,   0, 1.0};
Point(newp) = {-2, 3.5, 0, 1.0};
Point(newp) = {+2, 3.5, 0, 1.0};
Point(newp) = {+2, 0.5, 0, 1.0};
Point(newp) = {+5, 0.5, 0, 1.0};
Point(newp) = {+5, 6.5, 0, 1.0};
Point(newp) = {-5, 6.5, 0, 1.0};
Point(newp) = {-5, 0,   0, 1.0};

Point(newp) = {-1, 0.5, 0, 1.0};
Point(newp) = {+1, 0.5, 0, 1.0};
Point(newp) = {+1, 2.5, 0, 1.0};
Point(newp) = {-1, 2.5, 0, 1.0};

Point(newp) = {+6, 0.5, 0, 1.0};
Point(newp) = {+8, 0.5, 0, 1.0};
Point(newp) = {+6, 2.5, 0, 1.0};
Point(newp) = {+8, 2.5, 0, 1.0};

Point(newp) = {-50,  0, 0, 10.0};
Point(newp) = {+50,  0, 0, 10.0};
Point(newp) = {+50, 50, 0, 10.0};
Point(newp) = {-50, 50, 0, 10.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};
Line(13) = {13, 14};
Line(14) = {14, 16};
Line(15) = {16, 15};
Line(16) = {15, 13};
Line(17) = {17, 8};
Line(18) = {1, 18};
Line(19) = {18, 19};
Line(20) = {19, 20};
Line(21) = {20, 17};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};
Curve Loop(2) = {11, 12, 9, 10};
Plane Surface(2) = {2};
Curve Loop(3) = {15, 16, 13, 14};
Plane Surface(3) = {3};
Curve Loop(4) = {18, 19, 20, 21, 17, -7, -6, -5, -4, -3, -2, -1};
Plane Surface(4) = {4, 2, 3};

Physical Surface("steel", 22) = {1};
Physical Surface("coil_pos", 23) = {2};
Physical Surface("coil_neg", 24) = {3};
Physical Surface("air", 25) = {4};
Physical Curve("neumann", 26) = {17, 8, 18};
Physical Curve("dirichlet", 27) = {21, 20, 19};
Physical Curve("gap", 28) = {1, 2, 3, 4, 5, 6, 7};
