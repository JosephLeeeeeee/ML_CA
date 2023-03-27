rho_value = [0.1 10 100];
gamma_value = [0.01 0.05 0.1 0.3 0.4 0.5 0.7 1];

R = 1;
A = [0 1 0; 0 -0.875 -20; 0 0 -50];
B = [0; 0; 50];
H = zeros(2,1);
Q = eye(2);

figure(Name = "fix gamma change rho");
for i = 1:length(rho_value)
    gamma = 0.01;
    rho = rho_value(i);
    G = [1 0 0 ; 0 gamma 0];
    QQ = G'*Q*G;
    RR = H'*Q*H + rho.*R;
    NN = G'*Q*H;

    [K,S,E] = lqr(A,B,QQ,RR,NN);

    sys_ltf = ss(A,B,K,0);

    bode(sys_ltf);
    [Gm,Pm,Wcg,Wcp] = margin(sys_ltf);
    Pm;
    Wcp;
    hold on;
end
hold off;
legend("\rho = 0.1","\rho = 10","\rho = 100");
legend(Location="best");

figure(Name = "fix rho change gamma");
for i = 1:length(gamma_value)
    rho = 0.01;
    gamma = gamma_value(i);
    G = [1 0 0 ; 0 gamma 0];
    QQ = G'*Q*G;
    RR = H'*Q*H + rho.*R;
    NN = G'*Q*H;

    [K,S,E] = lqr(A,B,QQ,RR,NN);

    sys_ltf = ss(A,B,K,0);

    bode(sys_ltf);
    [num,den] = ss2tf(A,B,K,0);
    [Gm,Pm,Wcg,Wcp] = margin(sys_ltf);
    Wcp;
    hold on;
end
hold off;
legend("\gamma = 0.01","\gamma = 0.1","\gamma =1");
legend(Location="best");



rho = 0.1;
gamma = 1;
G = [1 0 0 ; 0 gamma 0];
QQ = G'*Q*G;
RR = H'*Q*H + rho.*R;
NN = G'*Q*H;
[K,S,E] = lqr(A,B,QQ,RR,NN);

figure(Name = "Nyquist without pole placement");
sys_ny = ss(A,B,K,0);
nyquist(sys_ny);
xlim([-2,2]);
ylim([-2,2]);


figure(Name = "Nyquist after pole placement");
K = place(A,B,[-1 -2 -3]);
sys_ny = ss(A,B,K,0);
nyquist(sys_ny);
xlim([-2,2]);
ylim([-2,2]);



R = 1;
A = [0 1 0; 0 -0.875 -20; 0 0 -50];
B = [0; 0; 50];
H = zeros(2,1);
Q = eye(2);

figure(Name = "55555");
for i = 1:length(rho_value)
    gamma = 0.01;
    rho = rho_value(i);
    G = [1 0 0 ; 0 gamma 0];
    QQ = G'*Q*G;
    RR = H'*Q*H + rho.*R;
    NN = G'*Q*H;

    [K,S,E] = lqr(A,B,QQ,RR,NN);
    
    A_cl = A-B*K;
    B_cl = [1; 0; 0];
    C_cl = [1 0 0];
    D_cl = 0;

    sys_cl = ss(A_cl, B_cl, C_cl, D_cl);
    step(sys_cl);
    hold on;
end
hold off;

figure(Name = "66666");
for i = 1:length(gamma_value)
    rho = 0.01;
    gamma = gamma_value(i);
    G = [1 0 0 ; 0 gamma 0];
    QQ = G'*Q*G;
    RR = H'*Q*H + rho.*R;
    NN = G'*Q*H;

    [K,S,E] = lqr(A,B,QQ,RR,NN);
    
    A_cl = A-B*K;
    B_cl = [1; 0; 0];
    C_cl = [1 0 0];
    D_cl = 0;

    sys_cl = ss(A_cl, B_cl, C_cl, D_cl);
    step(sys_cl);
    hold on;
end
