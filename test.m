clc;
clear;

A=ones(3);
B=[A*0.25,A*0.5;A*0.5,A];
C=B+eye(6)*1e-12;

[V1,D1]= eig(B);
[V2,D2]= eig(C);

V2*D2*V2'