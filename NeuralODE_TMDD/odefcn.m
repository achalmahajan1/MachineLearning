function dydt = odefcn(t,y)
    kel = 0.5230;
    kon = 0.0485;
    km = 0.0458;
    koff = 0.0138;
    kdeg = 0.0934;
    ksyn = 0.934; 
    dydt =[-kon*y(1)*y(3) + koff*y(2) + ksyn - kdeg*y(1);kon*y(1)*y(3) - koff*y(2) - km*y(2);-kon*y(1)*y(3) + koff*y(2) - kel*y(3)];
end