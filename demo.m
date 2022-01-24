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

G = Phi \ Gamma;


%%
%  Compute a minimal realization of G

coef_dom_n = zeros(dim);
coef_dom_d1 = zeros(dim);
coef_dom_d2 = zeros(dim);

for i=1:dim
    for j=1:dim
        temp = G(i,j);
        temp_x = temp.num{1};
        temp_y = temp.den{1};
        coef_dom_n(i,j) = temp_x(1);
        coef_dom_d1(i,j) = temp_y(1);
        coef_dom_d2(i,j) = temp_y(2);
    end
end

norm(coef_dom_d1) % is equal to 0
norm(coef_dom_d2 - ones(dim)) % is equal to 0
rank(coef_dom_n) % is equal to 1, while the argument is -0.2 * ones(dim)

% These imply that it has a simple pole at infinity, which we shall isolate

% Apply bilinear mapping (s->1/s) to the network's TFM, thus turning it 
% proper to allow separation into the origina's proper and improper parts

Gammaf = flip_TFM_ss(Gamma);
Phif = flip_TFM_ss(Phi);
Gf = balreal(ss(Phif \ Gammaf,'min')); % form remapped network

Gf_propl = Gf - tf(1,[1 0]) * coef_dom_n; % no poles at 0, and therefore
                                          % the unmapped TFM is proper
Gf_propl = balreal(ss(Gf_propl,'min'));
G_prop = flip_TFM_ss(Gf_propl);

G_improp = coef_dom_n(1,1)*ones(dim,1)*ss(tf([1,0],1))*ones(1,dim);
G_dss = G_prop + G_improp;

norm(G - G_dss, inf) % approximately 0

%%
%  Form NRCF of original network

lam_G = -1:-0.1:-1-0.1*(length(Gf.a)-1);
F = place_dss(G_dss.e, G_dss.a, G_dss.b,lam_G);
orig_RCF = dss(G_dss.a + G_dss.b * F,   G_dss.b, [F;...
              G_dss.c + G_dss.d * F], [eye(dim) ; G_dss.d], G_dss.e);
% form stable RCF for the spectral factorization problem
orig_RCF = balreal(ss(orig_RCF,'min')); % eliminate nondynamic modes

Ao = orig_RCF.a;
Bo = orig_RCF.b;
Co = orig_RCF.c;
Do = orig_RCF.d;
   
[Xo, ~, Fo] = care(Ao, Bo, Co' * Co, Do' * Do, Co' * Do);
               
orig_NRCF = ss(Ao - Bo * Fo, Bo / chol(Do' * Do), Co - Do * Fo,...
               Do / chol(Do' * Do));
orig_NRCF = balreal(ss(orig_NRCF,'min'));

%%
%  Form network approximation which allows for augmented sparsity

Gbar = (s + 4);
kappa = 2;
Tk = [zeros(1, dim); [2 * eye(dim - 1) zeros(dim - 1, 1)]] + eye(dim);
Tk(1, dim) = kappa;

%% 
%  Exploit approximation structure for computation of admissible feedbacks

Gbarss = ss(Gbar - evalfr(Gbar, 0)); % select descriptor feedthrough
Gbarss.d = evalfr(Gbar, 0);          % for increased numerical accuracy 
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
%  along with the NRCF of the approximated netwrok's TFM

temp_RCF = balreal(ss([Mb; Nb], 'min'));
appr_RCF = ss(diag_rep(temp_RCF.a, dim), diag_rep(temp_RCF.b, dim) * Tk,...
       [Tk \ diag_rep(temp_RCF.c(1, :), dim);...
       diag_rep(temp_RCF.c(2, :), dim)],...
       [Tk \ diag_rep(temp_RCF.d(1, :), dim);...
       diag_rep(temp_RCF.d(2, :), dim)] * Tk);
appr_RCF = balreal(ss(appr_RCF,'min'));

Ar = appr_RCF.a;
Br = appr_RCF.b;
Cr = appr_RCF.c;
Dr = appr_RCF.d;
   
[Xr, ~, Fr] = care(Ar, Br, Cr' * Cr, Dr' * Dr, Cr' * Dr);
Fr = - Fr;
Hr = chol(Dr'*Dr);

appr_NRCF = ss(Ar + Br * Fr, Br / Hr, Cr + Dr * Fr, Dr / Hr);
appr_NRCF = balreal(ss(appr_NRCF,'min'));

%% 
%  Check the numerical validity of the obtained DCF and NRCFs

% Form and check the DCF for the approximated network's TFM
LCF_d = ss([Tk \ Ytb * Tk Tk \ (- Xtb * eye(dim));...
         - Ntb * Tk (Mtb * eye(dim))], 'min');
RCF_d = ss([Tk \ (Mb * Tk) Tk \ (eye(dim) * Xb);...
         Nb * Tk eye(dim) * Yb], 'min');
norm(LCF_d * RCF_d - eye(dim * 2), inf)
% approximately 0

% Validate the approximation's RCF and NRCF
norm(Ga - appr_RCF(dim + 1:2 * dim, :) / appr_RCF(1:dim, :), inf)
% approximately 0
norm(Ga - appr_NRCF(dim + 1:2 * dim, :) / appr_NRCF(1:dim, :), inf)
% approximately 0
norm(appr_NRCF'*appr_NRCF - eye(dim), inf) % approximately 0

% Validate the original network's RCF and NRCF
norm(G - orig_RCF(dim + 1:2 * dim, :) / orig_RCF(1:dim, :), inf)
% approximately 0
norm(G - orig_NRCF(dim + 1:2 * dim, :) / orig_NRCF(1:dim, :), inf)
% approximately 0
norm(orig_NRCF'*orig_NRCF - eye(dim), inf)
% approximately 0

%% 
%  Compute maximum stability radius for the approximated network's TFM

Gc = lyap(appr_NRCF.a, appr_NRCF.b*appr_NRCF.b');
Go = lyap(appr_NRCF.a', appr_NRCF.c'*appr_NRCF.c);
max_rad = sqrt(1 - max(abs(eig(Gc * Go)))); % maximum stability radius
disp(max_rad) % large value indicates potential for good robustness

%%
%  Compute the smallest stability radius which needs to be ensured to cover
%  the distance in H-infinity norm between the approximated network's NRCF
%  and the Delta-perturbed version of the latter, which acts as an 
%  arbitrary right factorization of the original network's TFM

% Compute NLCF of the original network's TFM

L = place_dss(G_dss.e', G_dss.a', G_dss.c',lam_G)';
orig_LCF = dss(G_dss.a + L * G_dss.c, [G_dss.b + (L * G_dss.d) L],...
               G_dss.c, [G_dss.d eye(dim)],G_dss.e);
orig_LCF = balreal(ss(orig_LCF, 'min')); % eliminate nondynamic modes

% The elimination of nondynamic modes may leave a residual singular value
% of around 1e-15 in the gain at infinity of the "denominator" TFM and we
% proceed to elimitate it, for increased numerical accuracy
[U,Sig,V] = svd(orig_LCF(:, dim + 1:2 * dim).d);
Sig(end,end) = 0;
orig_LCF = (ss(orig_LCF.a, orig_LCF.b, orig_LCF.c,...
              [orig_LCF(:, 1:dim).d U*Sig*V']));

Al = orig_LCF.a';
Bl = orig_LCF.c';
Cl = orig_LCF.b';
Dl = orig_LCF.d';
   
[Xl, ~, Fl] = care(Al, Bl, Cl' * Cl, Dl' * Dl, Cl' * Dl);
Fl = - Fl;
Hl = chol(Dl' * Dl);

orig_NLCF = ss(Al + Bl * Fl, Bl / Hl, Cl + Dl * Fl, Dl / Hl);
orig_NLCF = ss(orig_NLCF.a', orig_NLCF.c', orig_NLCF.b', orig_NLCF.d');
orig_NLCF = balreal(ss(orig_NLCF, 'min'));

% Validate the original network's LCF and NLCF
norm(G - orig_LCF(:, dim + 1:2 * dim) \ orig_LCF(:, 1:dim), inf)
% approximately 0
norm(G - orig_NLCF(:, dim + 1:2 * dim) \ orig_NLCF(:, 1:dim), inf)
% approximately 0
norm(orig_NLCF * orig_NLCF' - eye(dim), inf)
% approximately 0

% Compute terms of the associated two-block distance problem
JJ = balreal(ss(orig_NLCF * blkdiag(-eye(dim), eye(dim)) * appr_NRCF,...
                'min'));
GG = balreal(ss(orig_NRCF' * appr_NRCF, 'min'));

dist_inf = norm(JJ, inf);
dist_sup = 1; % must be stricty larger than dist_inf to allow factorization
tol_dist = 1e-6; % absolute tolerance for the bound computed below

% Compute an upper bound for the directed gap metric through bisection
while dist_sup-dist_inf > tol_dist
    
    dist_check = (dist_sup + dist_inf)/2;
    
    % The following system is guaranteed to be positive-definite
    % on the extended imaginary axis
    FF = balreal(ss(dist_check^2*eye(dim)-JJ'*JJ,'min'));
    
    [sFF, aFF] = stabsep(FF - FF.d); % isolate the stable part
    Aff = sFF.a;
    Bff = sFF.b;
    Cff = sFF.c;

    Xff = care(Aff, Bff, zeros(size(Aff)), FF.d, Cff');
    % compute the spectral factor
    SF  = ss(Aff, Bff , sqrtm(FF.d) \ (Cff + Bff' * Xff), sqrtm(FF.d));
    SF = balreal(ss(SF,'min'));
    SFi = balreal(ss(inv(SF), 'min'));
    RR = balreal(ss(GG*SFi,'min')); % symbol of the desired Hankel operator
        
    if sum(eig(RR) >= 0) < 1
        Hankel_norm_sq = 0;
    else
        [~,anti_stab_RR] = stabsep(RR - RR.d);
        Lc = lyap(anti_stab_RR.a, anti_stab_RR.b*anti_stab_RR.b');
        Lo = lyap(anti_stab_RR.a', anti_stab_RR.c'*anti_stab_RR.c);
        % square of the norm for the Hankel operator of symbol RR
        Hankel_norm_sq = max(abs(eig(Lc * Lo)));
    end
    
    if Hankel_norm_sq < 1
        dist_sup = dist_check;
    else
        dist_inf = dist_check;
    end
    
end

disp(dist_sup) % just under 0.5609

%%
%  Setup iterative procedure

eps_m = 0.7; % desired stability margin which is greater than dist_sup
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
T_sys = balreal(ss(T_sys, 'min'));

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
%Con_d = dbar == d;

% Define constraints
Con = [Con_def, Con_implem, Con_norm, Con_X_pos];%, Con_d];

sol = optimize(Con, [], options); % check LMI feasibility (necessary)

% Analyze error flags
if sol.problem ~= 0
    disp('Hmm, something went wrong!');
    sol.info
    yalmiperror(sol.problem)
end

max_iter = 100; % maximum number of allowed iterations
resid = cell(1, max_iter); % bilinear constraint violation per iteration

% Compute post-initialization cost
resid{1} = sum(svd(value(T_C) - value(T_A) * value(T_B)));

while iter < max_iter && resid{iter + 1} > eps_safe
    
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
 
    resid{iter + 1} = sum(svd(value(T_C) - value(T_A) * value(T_B)));
    
    if resid{iter + 1} <= eps_safe
        break; % iteration converged after odd number of steps
    end
    
    % Prepare next iteration
    iter = iter + 1;
    Xkb = - value(T_A);
    Ykb = - value(T_B);
    dbar_prev = value(dbar);
    X_prev = value(X);
    
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
 
end

%% 
%  Obtain the Youla parameter which ensures NRF structure and robustness

Qs = tf(value(d),1) * eye(dim); % express the solution of the optimization 
Q = (Tk \ Qs * eye(dim)); % form the Youla parameter for the original DCF

K_LCF = balreal(ss(Tk \ (Ytb * Tk) + Q * Ntb * Tk, 'min')); % left factor
K_RCF = balreal(ss(Tk \ (Xtb * eye(dim)) + Q * Mtb, 'min')); % right factor
K = balreal(ss(K_LCF \ K_RCF, 'min')); % the controller's structureless TFM

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
          
T1e = balreal(ss(T1e, 'min')); % eliminate nondynamic modes


T2e = ss(Ar, Br, - eps_m * Hr * Fr, eps_m * Hr); % no nondynamic modes
T2e = balreal(ss(T2e,'min')); 

T3e = dss(Aa + La * Ca, [La, - Ba - La * Da], Ca, [eye(dim), - Da], Ea);
T3e = balreal(ss(T3e, 'min')); % eliminate nondynamic modes

T_CL_r = T1e + T2e * Q * T3e;
T_CL_r = balreal(ss(T_CL_r, 'min'));

rob_marg_1 = eps_m / hinfnorm(T_CL_r, inf); % greater than eps_m
disp(rob_marg_1) % display obtained stability radius for approximation

rob_marg_2 = ncfmargin(Ga, K, 1, 1e-6); % greater than dist_sup
disp(rob_marg_2) % validate previous computation

rob_marg_3 = ncfmargin(G, K, 1, 1e-6); % good robustness indicator
disp(rob_marg_3) % display obtained stability radius for original network

%% 
%  Form controller NRF

K_LF_struc = Tk* K_LCF; % form structured left  factor of controller LCF
K_RF_struc = Tk* K_RCF; % form structured right factor of controller LCF

Phi_K = tf(0, 1) * ones(dim);
Gamma_K = tf(0, 1) * ones(dim);

Phi_K(1, dim) = -balreal(ss(K_LF_struc(1, 1) \ K_LF_struc(1, dim), 'min'));
Gamma_K(1, 1) = balreal(ss(K_LF_struc(1, 1) \ K_RF_struc(1, 1), 'min'));

for i = 2:dim
  Phi_K(i, i - 1) = -balreal(ss(K_LF_struc(i, i) \ K_LF_struc(i, i - 1),...
                                 'min'));
  Gamma_K(i, i) = balreal(ss(K_LF_struc(i, i) \ K_RF_struc(i, i), 'min'));
end

%% 
%  Approximate solution

Phi_K_a = Phi_K; % preserve feedforward term
Gamma_K_a = floor(evalfr(Gamma_K,inf)); % approximate feedback term
K_a = (eye(dim) - Phi_K_a) \ Gamma_K_a; % form apprixmated controller's TFM

rob_marg_a = ncfmargin(G, K_a, 1,1e-6); % still a good robustness indicator
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

 	
%%

function [Gf] = flip_TFM_ss(G)

    % Auxiliary function which effects the change of variable s -> 1/s
    % and returns the state-space realization of the resulting system
    % This implementation assumes G has no poles at 0 or at infinity

    temp = ss(G);
    D = evalfr(temp, 0); % G(0) becomes G(Inf)
    Gft = balreal(ss((D - G) * tf(1, [1 0]), 'min')); % obtain realization
                                                      % for (G(0) - G(s))/s
                                                     
    % Note that when shifting from s to p = 1/s, we now have that
    % Gft.c * inv(I - p * Gft.a) * Gft.b * p = (D - G(p)) * p  
    
    Gf = balreal(ss(inv(Gft.a), Gft.a \ Gft.b, Gft.c, D)); % form G(1/s) 
    Gf = balreal(ss(Gf, 'min'));
    
end
