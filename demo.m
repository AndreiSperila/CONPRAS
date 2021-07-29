%% 
%  Cleanup and initialisation

clear;
clc;
close all;

%% 
%  Input network TFM

s = tf('s');

g1 = - (s + 2) / (s + 1);
g2 = (4 * s + 7) / (s - 1);

dim = 20;
Gamma = eye(dim) * g2;
Phi = eye(dim) * tf(1, 1);
Phi(1, dim) = g1;
for i = 2:dim
    Phi(i, i - 1) = g1;
end

G = tf(Phi \ Gamma);

%% 
%  Apply bilinear mapping to network TFM, turning it into a proper TFM
%  and preserving the gap metric with respect to other transformed systems

Gammaf = ss(eye(dim) * tf(flip(g2.num{1}), flip(g2.den{1})));
Phif = eye(dim) * tf(1, 1);
Phif (1, dim) = tf(flip(g1.num{1}), flip(g1.den{1}));
for i = 2:dim
    Phif (i, i - 1) = tf(flip(g1.num{1}), flip(g1.den{1}));
end
iPhif = ss(inv(Phif));
Gf = Gammaf * iPhif;

%%
%  Form network approximation which allows for augmented sparsity

Gbar = (s + 4);
kappa = 2;
Tk = [zeros(1, dim); [2 * eye(dim - 1) zeros(dim - 1, 1)]] + eye(dim);
Tk(1, dim) = kappa;

%%
%  Apply bilinear mapping to approximated network TFM to easily compute gap
%  metric between it and the transformed system

Gbarl = tf(flip(Gbar.num{1}), flip(Gbar.den{1}));
Gal = Gbarl * Tk;
net_dist = gapmetric(Gf, Gal, 1e-5); % force increased accuracy
disp(net_dist) % approximately 0.5609, unless Gf not accurately computed

%% 
%  Exploit approximation structure for computation of admissible feedbacks

Gbarss = ss(Gbar - evalfr(Gbar, 0)); % select descriptor feedthrough
Gbarss.d = evalfr(Gbar, 0);          % for increased accuracy 
A = Gbarss.a;
B = Gbarss.b;
C = Gbarss.c;
D = Gbarss.d;
E = Gbarss.e;

lam = - 4;
F = place_dss(E, A, B, lam);
L = place_dss(E',A', C',lam)';

%%
%  Compute auxiliary DCF

aux_LCF = tf(dss(A + L * C, [- B - (L * D) L], [F; C], [1 0; - D 1], E));
aux_LCF = tf(ss(aux_LCF, 'min')); % eliminate nondynamic modes
aux_RCF = tf(dss(A + B * F, [B, - L], [F; C + D * F], [1 0; D 1], E));
aux_RCF = tf(ss(aux_RCF, 'min')); % eliminate nondynamic modes

Ytb =   aux_LCF(1, 1);
Xtb = - aux_LCF(1, 2);
Ntb = - aux_LCF(2, 1);
Mtb =   aux_LCF(2, 2);
Mb  =   aux_RCF(1, 1);
Xb  =   aux_RCF(1, 2);
Nb  =   aux_RCF(2, 1);
Yb  =   aux_RCF(2, 2);

%%
%  Compute both the descriptor realization and the two admissible feedbacks
%  for the approximated network

Ga = Gbarss * Tk;
Aa = Ga.a;
Ba = Ga.b;
Ca = Ga.c;
Da = Ga.d;
Ea = Ga.e;

Fa = Tk \ diag_rep(F, dim);
La = diag_rep(L, dim);

%%
%  Compute a stable right coprime factorization for the approximated
%  network's TFM and use it to find its corresponding spectral factor

temp_RCF = ss([Mb; Nb], 'min');
RCF = ss(diag_rep(temp_RCF.a, dim),...
       diag_rep(temp_RCF.b, dim) * Tk,...
       [Tk \ diag_rep(temp_RCF.c(1, :), dim);...
       diag_rep(temp_RCF.c(2, :), dim)],...
       [Tk \ diag_rep(temp_RCF.d(1, :), dim);...
       diag_rep(temp_RCF.d(2, :), dim)] * Tk);
Ar = RCF.a;
Br = RCF.b;
Cr = RCF.c;
Dr = RCF.d;
   
[Xr, ~, Fr] = care(Ar, Br, Cr' * Cr,...
                   Dr' * Dr, Cr' * Dr);
Fr = - Fr;
Hr = chol(Dr'*Dr);

G0 = ss(Ar, Br, - Hr * Fr, Hr);
G0i = ss(Ar + Br * Fr, Br / Hr, Fr, inv(Hr));

%% 
%  Check the numerical validity of the obtained DCF and NRCF

% Apply bilinear mapping, which preserves L-infinity norm, for increased
% numerical precision
norm((Ga * (Tk * Mb) / Tk) - (Nb * Tk), inf) % approximately 0

% Form DCF for the approximated network's TFM
LCF_d = ss([Tk \ Ytb * Tk Tk \ (- Xtb * eye(dim));...
         - Ntb * Tk (Mtb * eye(dim))], 'min');
RCF_d = ss([Tk \ (Mb * Tk) Tk \ (eye(dim) * Xb);...
         Nb * Tk eye(dim) * Yb], 'min');
norm(ss(LCF_d * RCF_d, 'min') - eye(dim * 2), inf) % approximately 0

% Check RCF used in the spectral factorization
norm(Ga * RCF(1:dim, :) - RCF(dim + 1:2 * dim, :), inf) % approximately 0

% Form NRCF block-column
ortg = ss(RCF * G0i, 'min'); 

% Check first right coprimeness and then normalization
norm(Ga * ortg(1:dim, :) - ortg(dim + 1:2 * dim, :), inf) % approximately 0
norm(ss(ortg'*ortg,'min')-eye(dim),inf) % approximately 0

%% 
%  Compute maximum stability radius for the approximated network's TFM

Gc = lyap(ortg.a',ortg.b*ortg.b'); % controllability gramian of NRCF
Go = lyap(ortg.a, ortg.c'*ortg.c); % observability gramian of NRCF
max_rad = sqrt(1 - max(abs(eig(Gc * Go)))); % maximum stability radius
disp(max_rad) % large value indicates potential for good robustness

%%
%  Setup interative procedure

eps_m = 0.7; % desired stability margin which is greater than net_dist
eps_safe = 1e-6; % imposed tolerance level

% Compute feedthrough controller "denominator"
Y_inf  = evalfr(Ytb, inf) * Tk;
N_inf  = evalfr(Ntb, inf) * Tk;
Yb_inf = evalfr(Yb, inf);
Nb_inf = evalfr(Nb, inf);

% Form the fixed part of the closed-loop system
T_sys = dss([Ar, - Br * Fa; zeros(length(Aa), length(Ar))...
             Aa + La * Ca], [[zeros(length(Ar), dim), - Br; La, - Ba...
             - La * Da] [Br / Tk; zeros(length(Aa), dim)]], [eps_m *...
             [- Hr * Fr, - Hr * Fa]; [zeros(dim, length(Ar)) Ca]], ...
             [[zeros(dim), - eps_m * Hr] eps_m * Hr / Tk;...
             [eye(dim), - Da] zeros(dim)], blkdiag(eye(size(Ar)), Ea));
T_sys = ss(T_sys, 'min');
T_sys = balreal(T_sys);

% Declare state matrices for fixed part of the closed-loop system
Af   = T_sys.a;
Bf1  = T_sys.b(:, 1:2 * dim);
Bf2  = T_sys.b(:, 2 * dim + 1:end);
Cf1  = T_sys.c(1:dim, :);
Cf2  = T_sys.c(dim + 1:2 * dim, :);
Df11 = T_sys.d(1:dim, 1:2 * dim);
Df12 = T_sys.d(1:dim, 2 * dim + 1:end);
Df21 = T_sys.d(dim + 1:end, 1:2 * dim);

iter = 0; % iteration number

%%
%  Define optimization problem through YALMIP

% Recommended solver for minimizing runtime is MOSEK
options = sdpsettings('verbose', 1, 'solver', 'mosek');


% Define decision variables
d = sdpvar(1);  % Force the free and stable parameter to be scalar and to
                % have identical entries for a homogenous control law
dbar = sdpvar(1); % Define auxiliary variable to prevent fixing d
d2 = sdpvar(1); % Auxiliary variable assigned to d * dbar
X = sdpvar(length(T_sys.a)); % Positive-definite matrix for norm bound
Xbar = sdpvar(length(T_sys.a)); % Auxiliary variable assigned to X

% Bounded real lemma matrix
sysmat = [(Af'*X+(Bf2*Cf2)' * Xbar) + (Af'*X+(Bf2*Cf2)' * Xbar)'...
          X*Bf1+Xbar*Bf2*Df21 (Cf1+Df12*Cf2*d)'; (X * Bf1 + Xbar * Bf2 *...
          Df21)' -eye(size(Df11' * Df11)) (Df11 + Df12 * Df21 * d)';...
          Cf1 + Df12 * Cf2 * d, Df11 + Df12 * Df21 * d,...
          - eye(size(Df11 * Df11'))];

% Bilinear optimization variables
T_C = blkdiag(Xbar, d2, d - dbar);
T_A = blkdiag(X, dbar, 0);
T_B = blkdiag(eye(size(X)) * d, d, 0);
[~, ma] = size(T_A);
[pm, mm] = size(T_C);

% Constraint for well-defined controller
Con_def = Y_inf'*Y_inf + [(Y_inf' * N_inf) (N_inf'*Y_inf)...
          (N_inf' * N_inf)] * [dbar * eye(dim); d * eye(dim);...
          d2 * (eye(dim))] >= eps_safe * eye(dim);

% Constraint for implementable NRF
Con_implem = Yb_inf'*Yb_inf + [(Yb_inf' * Nb_inf) (Nb_inf'*Yb_inf)...
             (Nb_inf' * Nb_inf)] * [dbar; d; d2] >= eps_safe * eye(1);

% Constraints for norm bound
Con_norm = sysmat <= - eps_safe * eye(size(sysmat));
Con_X_pos = X >= eps_safe * eye(size(X));

% Initial constraint for free term (optional, but speeds up convergence)
Con_d = dbar == d;

% Define constraints
Con = [Con_def, Con_implem, Con_norm, Con_X_pos, Con_d];

sol = optimize(Con, [], options); % check LMI feasibility (necessary)

% Analyze error flags
if sol.problem ~= 0
    disp('Hmm, something went wrong!');
    sol.info
    yalmiperror(sol.problem)
end

max_iter = 20; % maximum number of allowed iterations
resid = cell(1, max_iter); % bilinear constraint violation per iteration

% Prepare first iteration
resid{1} = norm(value(T_C) - value(T_A) * value(T_B), 'fro');
Xkb = - value(T_A);
Ykb = - value(T_B);
dbar_prev = value(dbar);
X_prev = value(X);

while iter < max_iter && resid{iter + 1} > eps_safe
 
    iter = iter + 1;
    
    % Update variables
    d = sdpvar(1);
    dbar = dbar_prev;
    d2 = sdpvar(1);
    X = X_prev;
    Xbar = sdpvar(length(T_sys.a));

    % Bounded real lemma matrix
    sysmat = [(Af'*X+(Bf2*Cf2)' * Xbar) + (Af'*X+(Bf2*Cf2)' * Xbar)'...
              X*Bf1+Xbar*Bf2*Df21 (Cf1+Df12*Cf2*d)'; (X * Bf1 + Xbar *...
              Bf2 * Df21)' -eye(size(Df11' * Df11)) (Df11 + Df12 * Df21...
              * d)'; Cf1 + Df12 * Cf2 * d, Df11 + Df12 * Df21 * d,...
              - eye(size(Df11 * Df11'))];

    % Bilinear optimization variables
    T_C = blkdiag(Xbar, d2, d - dbar);
    T_A = blkdiag(X, dbar, 0);
    T_B = blkdiag(eye(size(X)) * d, d, 0);
    [~, ma] = size(T_A);
    [pm, mm] = size(T_C);

    % Constraint for well-defined controller
    Con_def = Y_inf'*Y_inf + [(Y_inf' * N_inf) (N_inf'*Y_inf)...
              (N_inf' * N_inf)] * [dbar * eye(dim); d * eye(dim);...
              d2 * (eye(dim))] >= eps_safe * eye(dim);

    % Constraint for implementable NRF
    Con_implem = Yb_inf'*Yb_inf + [(Yb_inf' * Nb_inf) (Nb_inf'*Yb_inf)...
                 (Nb_inf' * Nb_inf)] * [dbar; d; d2] >= eps_safe * eye(1);

    % Constraints for norm bound
    Con_norm = sysmat <= - eps_safe * eye(size(sysmat)); % X is fixed
    
    % Define constraints
    Con = [Con_def, Con_implem, Con_norm];

    Obj = norm([T_C + Xkb * T_B zeros(pm, ma);...
                T_B + Ykb eye(ma)], 'nuclear'); % nuclear norm relaxation
 
    sol = optimize(Con, Obj, options);
 
    % Analyze error flags
    if sol.problem ~= 0
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end
 
    resid{iter + 1} = sum(svd(value(T_C) - value(T_A) * value(T_B)));
 
    if resid{iter + 1} <= eps_safe
        break; % iteration converged after odd number of steps
    end
    
    % Prepare next iteration
    iter = iter + 1;
    Xka = - value(T_A);
    Yka = - value(T_B);
    d_prev = value(d);
    
    % Update variables
    d = d_prev;
    dbar = sdpvar(1);
    d2 = sdpvar(1);
    X = sdpvar(length(T_sys.a));
    Xbar = sdpvar(length(T_sys.a));

    % Bounded real lemma matrix
    sysmat = [(Af'*X+(Bf2*Cf2)' * Xbar) + (Af'*X+(Bf2*Cf2)' * Xbar)'...
              X*Bf1+Xbar*Bf2*Df21 (Cf1+Df12*Cf2*d)'; (X * Bf1 + Xbar *...
              Bf2 * Df21)' -eye(size(Df11' * Df11)) (Df11 + Df12 * Df21...
              * d)'; Cf1 + Df12 * Cf2 * d, Df11 + Df12 * Df21 * d,...
              - eye(size(Df11 * Df11'))];

    % Bilinear optimization variables
    T_C = blkdiag(Xbar, d2, d - dbar);
    T_A = blkdiag(X, dbar, 0);
    T_B = blkdiag(eye(size(X)) * d, d, 0);
    [~, ma] = size(T_A);
    [pm, mm] = size(T_C);



    % Constraint for well-defined controller
    Con_def = Y_inf'*Y_inf + [(Y_inf' * N_inf) (N_inf'*Y_inf)...
              (N_inf' * N_inf)] * [dbar * eye(dim); d * eye(dim);...
              d2 * (eye(dim))] >= eps_safe * eye(dim);

    % Constraint for implementable NRF
    Con_implem = Yb_inf'*Yb_inf + [(Yb_inf' * Nb_inf) (Nb_inf'*Yb_inf)...
                 (Nb_inf' * Nb_inf)] * [dbar; d; d2] >= eps_safe * eye(1);

    % Constraints for norm bound
    Con_norm = sysmat <= - eps_safe * eye(size(sysmat));
    Con_X_pos = X >= eps_safe * eye(size(X));
    
    % Define constraints
    Con = [Con_def, Con_implem, Con_norm, Con_X_pos];

    Obj = norm([T_C + T_A * Yka T_A + Xka;...
              zeros(ma, mm) eye(ma)], 'nuclear'); % nuclear norm relaxation
 
    sol = optimize(Con, Obj, options);
 
    % Analyze error flags
    if sol.problem ~= 0
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end
    
    % Prepare next iteration
    Xkb = - value(T_A);
    Ykb = - value(T_B);
    dbar_prev = value(dbar);
    X_prev = value(X);
 
    resid{iter + 1} = sum(svd(value(T_C) - value(T_A) * value(T_B)));
 
end

%% 
%  Obtain the Youla parameter which ensures NRF structure and robustness

Qs = tf(value(d),1) * eye(dim); % express the solution of the optimization 
Q = (Tk \ Qs * eye(dim)); % form the Youla parameter for the original DCF

K_LCF = ss(Tk \ (Ytb * Tk) + Q * Ntb * Tk, 'min'); % left coprime factor
K_RCF = ss(Tk \ (Xtb * eye(dim)) + Q * Mtb, 'min'); % right coprime factor
K = ss(K_LCF \ K_RCF, 'min'); % the controller's structureless TFM

%% 
%  Form closed-loop system

T_RS_ss = dss([Ar, - Br * Fa; zeros(length(Aa), length(Ar)) Aa], ...
              [zeros(length(Ar), dim), - Br, Br; zeros(length(Aa), dim),...
              - Ba, Ba], [- eps_m * Hr * [Fr Fa]; zeros(size(Fr)) Ca],...
              [zeros(dim), - eps_m * Hr, eps_m * Hr; eye(dim), - Da,...
              Da], blkdiag(eye(size(Ar)), Ea));

T1e = dss([Ar, - Br * Fa; zeros(length(Aa), length(Ar)) Aa + La * Ca], ...
          [zeros(length(Ar), dim), - Br; La, - Ba - La * Da], eps_m * ...
          [- Hr * Fr, - Hr * Fa], [zeros(dim), - eps_m * Hr],...
          blkdiag(eye(size(Ar)), Ea));
T1e = ss(T1e, 'min'); % eliminate nondynamic modes

T2e = ss(Ar, Br, - eps_m * Hr * Fr, eps_m * Hr); % no nondynamic modes

T3e = dss(Aa + La * Ca, [La, - Ba - La * Da], Ca, [eye(dim), - Da], Ea);
T3e = ss(T3e, 'min'); % eliminate nondynamic modes

T_CL_r = T1e + T2e * Q * T3e;
T_CL_r = ss(T_CL_r, 'min');

rob_marg_1 = eps_m / hinfnorm(T_CL_r, inf); % greater than eps_m
disp(rob_marg_1) % display obtained stability radius for approximation

rob_marg_2 = ncfmargin(Ga, K, 1); % greater than net_dist
disp(rob_marg_2) % validate previous computation

rob_marg_3 = ncfmargin(G, K, 1); % good robustness indicator
disp(rob_marg_3) % display obtained stability radius for original network

%% 
%  Form controller NRF

K_LF_struc = Tk* K_LCF; % form structured left  factor of controller LCF
K_RF_struc = Tk* K_RCF; % form structured right factor of controller LCF

Phi_K = tf(0, 1) * ones(dim);
Gamma_K = tf(0, 1) * ones(dim);

Phi_K(1, dim) = - ss(K_LF_struc(1, 1) \ K_LF_struc(1, dim), 'min');
Gamma_K(1, 1) = ss(K_LF_struc(1, 1) \ K_RF_struc(1, 1), 'min');
for i = 2:dim
  Phi_K(i, i - 1) = - ss(K_LF_struc(i, i) \ K_LF_struc(i, i - 1), 'min');
  Gamma_K(i, i) = ss(K_LF_struc(i, i) \ K_RF_struc(i, i), 'min');
end

%% 
%  Approximate solution

Phi_K_a = Phi_K; % preserve feedforward term
Gamma_K_a = floor(evalfr(Gamma_K,inf)); % approximate feedback term
K_a = (eye(dim) - Phi_K_a) \ Gamma_K_a; % form apprixmated controller's TFM

rob_marg_a = ncfmargin(G, K_a, 1); % still a good robustness indicator
disp(rob_marg_a) % display obtained stability radius after approximation

%%

 	
function F = place_dss(E, A, B, lam_vec)

    % Auxiliary function for generalized eigenvalue placement via 
    % static "state" feedback for descriptor systems
    % Steps 1-5 implemented from "On stabilization methods of descriptor 
    % systems", A. Varga, Systems & Control Letters 24 (1995), pp. 133-138
    % Steps 6-7 implemented with explicit inversion of the E-matrix due to 
    % lack of numerical concerns in the appoximated network's TFM

    n = length(E);

    % Step 1: bring descriptor realization to SVD coordinate form
    
    [U_E, S_E, V_E] = svd(E);
    U_E = U_E';
    re = rank(S_E);

    flip_mat_1 = flip(eye(n));
    E = flip_mat_1 * S_E * flip_mat_1;
    U_E = flip_mat_1 * U_E;
    V_E = V_E * flip_mat_1;

    A = U_E * A * V_E;
    B = U_E * B;
    B_1 = B(1:n - re, :);
    B_2 = B(n - re + 1:end, :);

    % Step 2: row-compress top part of input matrix

    [nb1, m] = size(B_1);
    [U_B, B_rc] = qr(B_1);
    U_B = U_B';
    rb1 = rank(B_rc);

    flip_mat_2 = flip(eye(nb1));
    U_B = flip_mat_2 * U_B;
    B_rc = flip_mat_2 * B_rc;

    B = [B_rc; B_2];
    B_21 = B_rc(nb1 - rb1 + 1:end, :);
    A = [U_B zeros(nb1, n - nb1); zeros(n - nb1, nb1) eye(n - nb1)] * A;

    % Step 3: column-compress top-left part of state matrix

    [V_A, R] = qr((A(1:nb1 - rb1, 1:n - nb1))');
    V_A = V_A';
    R = R';
    na = rank(R);
    if na > 0 % check doubles as validation of impulse controllabilitiy
        A = A * [V_A zeros(n - re, re); zeros(re, n - re) eye(re)];
    end

    % Step 4: find feedback matrix which renders the pencil impulse-free

    F_12 = pinv(B_21) * (eye(nb1 - na) - A(na + 1:n - re, na + 1:n - re));
    F_1 = [zeros(m, na) F_12 zeros(m, re)];
    A = A + B * F_1;

    % Step 5: bring the pole pencil to block-upper triangular form

    [U_F, ~] = qr(A(:, 1:n - re));
    U_F = U_F';
    E = U_F * E;
    A = U_F * A;
    B = U_F * B;

    E_22 = E(n - re + 1:end, n - re + 1:end);
    A_22 = A(n - re + 1:end, n - re + 1:end);
    B_2 = B(n - re + 1:end, :);

    % Step 6: allocate the generalized eigenvalues of (A_22+B_2*F_22,E_22)

    temp = ss(E_22 \ A_22, E_22 \ B_2, eye(re), zeros(re, m));
    [tempb, ~, T] = balreal(temp); % balance the realization to avoid
                                   % even mild numerical concerns
    F_22 = - place(tempb.a, tempb.b, lam_vec);
    F_22 = F_22 * T;
    F_2 = [zeros(m, n - re) F_22];
    
    % Step 7: form the final feedback matrix
    
    if na > 0
        F = (F_1 + F_2) * [V_A' zeros(n-re,re);...
                                zeros(re,n-re) eye(re)]*V_E';
    else
        F = (F_1 + F_2) * V_E';
    end

end

%%
 	
function Xt = diag_rep(X, n)
    
    % Auxiliary function for the block-diagonal repetition of a matrix

    Xt = [];
    
    for i = 1:n
        Xt = blkdiag(Xt, X);
    end
    
end
