%%
%  Cleanup and initialisation

clear;
clc;
close all;

%%
%  Method selection

%  1 - the procedure proposed in [1] and based upon Algorithm 1 in [2]

%  2 - the procedure based upon Algorithm 2 in [2]

%  3 - the procedure based upon Algorithm 1 in [3]

%  4 - the procedure based upon Algorithm 1 in [4]

method = 1;

% [1] - A. Sperilă, C. Oară, B. D. Ciubotaru, and Ş. Sabău, “Distributed
%       Control of Descriptor Networks: A Convex Procedure for Augmented
%       Sparsity," pp. 1–8, 2021, [Online].
%       Available: https://arxiv.org/abs/2109.05954.

% [2] - R. Doelman and M. Verhaegen, “Sequential convex relaxation for
%       convex optimization with bilinear matrix equalities," in 2016
%       European Control Conference (ECC), 2016, pp. 1946–1951.

% [3] - C. Sun and R. Dai, “A customized ADMM for rank-constrained
%       optimization problems with approximate formulations," In Proc. of
%       the IEEE 56th Conference on Decision and Control, pp. 3769–3774,
%       2017.

% [4] - C. Sun and R. Dai, “Rank-constrained optimization and its
%       applications," Automatica, vol. 82, pp. 128–136, 2017.

%%
%  Input network TFM

s = tf('s');

G_bar_y = (s + 2) / (s + 1);
G_bar_u = (4 * s + 7) / (s - 1);

dim = 20;
G_bar_D = eye(dim) * G_bar_u;
G_bar_S = eye(dim) * tf(1, 1);
G_bar_S(1, dim) = - G_bar_y;

for i = 2:dim
    G_bar_S(i, i - 1) = - G_bar_y;
end

G_bar = G_bar_S \ G_bar_D;

%%
%  Obtain a realization for the network's model

G_bar_y_dss = dss(blkdiag(-1, -1),  [1; 1], [1 1], 0, [1 0; 0 0]);
G_bar_u_dss = dss(blkdiag( 1, -1), [11; 4], [1 1], 0, [1 0; 0 0]);

Xi_1 = [zeros(1, dim-1) 1; eye(dim-1) zeros(dim-1, 1)];
Xi_bar = diag_rep(G_bar_y_dss.b, dim) * Xi_1;

A_G_bar = [diag_rep(G_bar_y_dss.a, dim) + Xi_bar * ...
    diag_rep(G_bar_y_dss.c, dim) Xi_bar * diag_rep(G_bar_u_dss.c, dim); ...
    zeros(2*dim) diag_rep(G_bar_u_dss.a, dim)];

B_G_bar = [zeros(2*dim, dim); diag_rep(G_bar_u_dss.b, dim)];

C_G_bar = [diag_rep(G_bar_y_dss.c, dim) diag_rep(G_bar_u_dss.c, dim)];

E_G_bar = blkdiag(diag_rep(G_bar_y_dss.e,dim),diag_rep(G_bar_u_dss.e,dim));

G_bar_dss = dss(A_G_bar, B_G_bar, C_G_bar, zeros(dim), E_G_bar);
dim_dss = length(A_G_bar);
norm(G_bar - G_bar_dss, inf) % approximately 0

% Verify that the original network's realization is both strongly
% stabilizable and strongly detectable
fin_dyn_stab = rank([A_G_bar  - 1 * E_G_bar  B_G_bar ]) == length(A_G_bar);
fin_dyn_dect = rank([A_G_bar' - 1 * E_G_bar' C_G_bar']) == length(A_G_bar);
impulse_ctrb = rank([E_G_bar  A_G_bar  * null(E_G_bar )  B_G_bar ]) == ...
    length(A_G_bar);
impulse_obsv = rank([E_G_bar' A_G_bar' * null(E_G_bar')  C_G_bar']) == ...
    length(A_G_bar);
prod(double([fin_dyn_stab fin_dyn_dect impulse_ctrb impulse_obsv]))
% output is 1 which indicates strong stabilizability and detectability

%%
%  Eliminate nondynamic modes from the obtained realization

% Bring system to SVD-like coordinate form
[Ue, Se, Ve] = svd(E_G_bar);
Re = rank(Se);
G_bar_dss = dss(Ue' * A_G_bar * Ve, Ue' * B_G_bar, C_G_bar * Ve, ...
    zeros(dim), Ue' * E_G_bar * Ve);
norm(G_bar - G_bar_dss, inf) % approximately 0

% Compress now the lower right-hand corner of the state matrix
[Ua, Sa, Va] = svd(G_bar_dss.a(Re+1 : end, Re+1 : end));
Ra = rank(Sa);
dif = dim_dss - Re - Ra;
Ua = blkdiag(eye(Re), Ua * flip(eye(dim_dss-Re))); % do the compression to
Va = blkdiag(eye(Re), Va * flip(eye(dim_dss-Re))); % the lower-right corner
G_bar_dss = dss(Ua'*G_bar_dss.a*Va, Ua'*G_bar_dss.b, G_bar_dss.c*Va, ...
    zeros(dim), Ua'*G_bar_dss.e*Va);
norm(G_bar - G_bar_dss, inf) % approximately 0

% Eliminate nondynamic part
A_red = G_bar_dss.a(1:Re+dif, 1:Re+dif) - ...
    (G_bar_dss.a(1:Re+dif, Re+dif+1:end) / ...
    G_bar_dss.a(Re+dif+1:end, Re+dif+1:end)) * ...
    G_bar_dss.a(Re+dif+1:end, 1:Re+dif);

B_red = G_bar_dss.b(1:Re+dif,:) - (G_bar_dss.a(1:Re+dif, Re+dif+1:end) /...
    G_bar_dss.a(Re+dif+1:end, Re+dif+1:end)) * G_bar_dss.b(Re+dif+1:end,:);

C_red = G_bar_dss.c(:, 1:Re+dif) - G_bar_dss.c(:, Re+dif+1:end) * ...
    (G_bar_dss.a(Re+dif+1:end, Re+dif+1:end) \ ...
    G_bar_dss.a(Re+dif+1:end, 1:Re+dif));

D_red = - G_bar_dss.c(:, Re+dif+1:end) * ...
   (G_bar_dss.a(Re+dif+1:end, Re+dif+1:end) \ G_bar_dss.b(Re+dif+1:end,:));

E_red = G_bar_dss.e(1:Re+dif, 1:Re+dif);

% Form the original network's realization that will be used in the sequel
G_bar_dss = dss(A_red, B_red, C_red, D_red, E_red);
norm(G_bar - G_bar_dss, inf) % approximately 0

% Verify that computational errors in nondynamic mode elimination din not
% numerically compromise the operation
double(rank([E_red A_red*null(E_red)]) == rank(E_red))
% output is 1 which indicates no nondynamic modes

% Verify that the matrix operations which eliminated the nondynamic part
% did not numerically compomise the reduced realization's strong
% stabilizability
fin_dyn_stab_red = rank([A_red  - 1 * E_red  B_red ]) == length(A_red);
impulse_ctrb_red = rank([E_red  A_red  * null(E_red )  B_red ]) == ...
    length(A_red);
prod(double([fin_dyn_stab impulse_ctrb]))
% output is 1 which indicates strong stabilizability

%%
%  Form NRCF of original network

lam_G = -1:-0.1:-1 - 0.1 * (length(G_bar_dss.a) - 2);
F = place_dss(G_bar_dss.e, G_bar_dss.a, G_bar_dss.b, lam_G);
orig_RCF = dss(G_bar_dss.a + G_bar_dss.b * F, G_bar_dss.b, [F; ...
    G_bar_dss.c + G_bar_dss.d * F], [eye(dim); G_bar_dss.d], G_bar_dss.e);
% form stable RCF for the spectral factorization problem
orig_RCF = balreal(ss(orig_RCF, 'min')); % eliminate nondynamic modes

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

Psi = (s + 4);
kappa = 2;
Tk = kappa * Xi_1 + eye(dim);

%%
%  Exploit approximation structure for computation of admissible feedbacks

Psi_dss = ss(Psi - evalfr(Psi, 0)); % select descriptor feedthrough
Psi_dss.d = evalfr(Psi, 0);         % for increased numerical accuracy
A = Psi_dss.a;
B = Psi_dss.b;
C = Psi_dss.c;
D = Psi_dss.d;
E = Psi_dss.e;
lam = -4;
F = place_dss(E, A, B, lam);
L = place_dss(E', A', C', lam)';

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

G = Psi_dss * Tk;
Aa = G.a;
Ba = G.b;
Ca = G.c;
Da = G.d;
Ea = G.e;
Fa = Tk \ diag_rep(F, dim);
La = diag_rep(L, dim);

%%
%  Compute a stable right coprime factorization for the approximated
%  network's TFM and use it to find its corresponding spectral factor
%  along with the NRCF of the approximated network's TFM

temp_RCF = balreal(ss([Mb; Nb], 'min'));
appr_RCF = ss(diag_rep(temp_RCF.a, dim), diag_rep(temp_RCF.b, dim) * Tk,...
    [Tk \ diag_rep(temp_RCF.c(1, :), dim);...
    diag_rep(temp_RCF.c(2, :), dim)],...
    [Tk \ diag_rep(temp_RCF.d(1, :), dim);...
    diag_rep(temp_RCF.d(2, :), dim)] * Tk);
appr_RCF = balreal(ss(appr_RCF,'min'));

% Realization to be used for spectral factorization with Er = eye(size(Ar))
Ar = appr_RCF.a;
Br = appr_RCF.b;
Cr = appr_RCF.c;
Dr = appr_RCF.d;
[Xr, ~, Fr] = care(Ar, Br, Cr' * Cr, Dr' * Dr, Cr' * Dr);
Fr = - Fr;
Hr = chol(Dr'*Dr);
appr_NRCF = ss(Ar + Br * Fr, Br / Hr, Cr + Dr * Fr, Dr / Hr);
appr_NRCF = balreal(ss(appr_NRCF, 'min'));

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
norm(G - appr_RCF(dim + 1:2 * dim, :) / appr_RCF(1:dim, :), inf)
% approximately 0
norm(G - appr_NRCF(dim + 1:2 * dim, :) / appr_NRCF(1:dim, :), inf)
% approximately 0
norm(appr_NRCF'*appr_NRCF - eye(dim), inf) % approximately 0

% Validate the original network's RCF and NRCF
norm(G_bar - orig_RCF(dim + 1:2 * dim, :) / orig_RCF(1:dim, :), inf)
% approximately 0
norm(G_bar - orig_NRCF(dim + 1:2 * dim, :) / orig_NRCF(1:dim, :), inf)
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

L = place_dss(G_bar_dss.e', G_bar_dss.a', G_bar_dss.c',lam_G)';
orig_LCF = dss(G_bar_dss.a + L * G_bar_dss.c, [G_bar_dss.b + ...
    (L * G_bar_dss.d) L], G_bar_dss.c, [G_bar_dss.d eye(dim)],G_bar_dss.e);
orig_LCF = balreal(ss(orig_LCF, 'min')); % eliminate nondynamic modes

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
norm(G_bar - orig_LCF(:, dim + 1:2 * dim) \ orig_LCF(:, 1:dim), inf)
% approximately 0
norm(G_bar - orig_NLCF(:, dim + 1:2 * dim) \ orig_NLCF(:, 1:dim), inf)
% approximately 0
norm(orig_NLCF * orig_NLCF' - eye(dim), inf)
% approximately 0

%%
% Compute terms of the associated two-block distance problem
JJ = balreal(ss(orig_NLCF * blkdiag(-eye(dim), eye(dim)) * appr_NRCF,...
    'min'));
GG = balreal(ss(orig_NRCF' * appr_NRCF, 'min'));

dist_inf = norm(JJ, inf);
dist_sup = 1; % must be stricty larger than dist_inf to allow factorization
tol_dist = 1e-6; % absolute tolerance for the bound computed below

% Compute an upper bound for the directed gap metric through bisection
while dist_sup - dist_inf > tol_dist

    dist_check = (dist_sup + dist_inf) / 2;

    % The following system is guaranteed to be positive-definite
    % on the extended imaginary axis
    FF = balreal(ss(dist_check^2*eye(dim)-JJ'*JJ,'min'));

    [sFF, aFF] = stabsep(FF - FF.d); % isolate the stable part
    Aff = sFF.a;
    Bff = sFF.b;
    Cff = sFF.c;

    Xff = care(Aff, Bff, zeros(size(Aff)), FF.d, Cff');
    % compute the spectral factor
    SF = ss(Aff, Bff, sqrtm(FF.d) \ (Cff + Bff' * Xff), sqrtm(FF.d));
    SF = balreal(ss(SF, 'min'));
    SFi = balreal(ss(inv(SF), 'min'));
    RR = balreal(ss(GG*SFi, 'min')); %symbol of the desired Hankel operator

    if sum(eig(RR) >= 0) < 1
        Hankel_norm_sq = 0;
    else
        [~,anti_stab_RR] = stabsep(RR - RR.d);
        Lc = lyap(anti_stab_RR.a, -anti_stab_RR.b*anti_stab_RR.b');
        Lo = lyap(anti_stab_RR.a', -anti_stab_RR.c'*anti_stab_RR.c);
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

% Compute feedthrough of controller "denominator"
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

tic % measure time spent in optimization procedure

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
sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + (Bf2 * Cf2)' * ...
    Xbar)', X * Bf1 + Xbar * Bf2 * Df21, (Cf1 + Df12 * Cf2 * d)'; ...
    (X * Bf1 + Xbar * Bf2 * Df21)', -eye(size(Df11'*Df11)), ...
    (Df11 + Df12 * Df21 * d)'; Cf1 + Df12 * Cf2 * d, Df11 + Df12 * ...
    Df21 * d, - eye(size(Df11*Df11'))];

% Bilinear optimization variables
T_C = blkdiag(Xbar, d2, d - dbar);
T_A = blkdiag(X, dbar, 0);
T_B = blkdiag(eye(size(X)) * d, d, 0);
[~, ma] = size(T_A);
[pm, mm] = size(T_C);

% Constraint for well-defined controller
Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), (N_inf' * Y_inf), ...
    (N_inf' * N_inf)] * [d * eye(dim); dbar * eye(dim); ...
    d2 * (eye(dim))] >= eps_safe * eye(dim);

% Constraint for implementable NRF
Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), (Nb_inf' * Yb_inf),...
    (Nb_inf' * Nb_inf)] * [d; dbar; d2] >= eps_safe * eye(1);

% Constraints for norm bound
Con_norm = sysmat <= -eps_safe * eye(size(sysmat));
Con_X_pos = X >= eps_safe * eye(size(X));

% Initial constraint for free term (optional, but speeds up convergence)
Con_d = dbar == d;

% Group up constraints
Con = [Con_def, Con_implem, Con_norm, Con_X_pos, Con_d];

if method == 4 % special intialization for the IRM-based method
    
    % Trace heuristic minimization variables
    M = [T_C, T_A; T_B, eye(ma)];
    [pm, mm] = size(M);
    Y = sdpvar(pm);
    Z = sdpvar(mm);

    Con_met_4 = [Con, ...
        [Y, M; M', Z] >= zeros(pm+mm), ... % classical trace heuristic
                                       ... % inequality constaint
        [eye(pm), M; M', Z] >= zeros(pm+mm)]; % enforce initialization
                                              % positive semidefiniteness

    sol = optimize(Con_met_4, trace(Y)+trace(Z), options); % solve for
                                                           % initialization

else

    sol = optimize(Con, [], options); % check LMI feasibility (necessary)

end

% Analyze error flags
if sol.problem ~= 0
    disp('Hmm, something went wrong!');
    sol.info
    yalmiperror(sol.problem)
end

if method == 4 % initialization setup for IRM-based mathod
    
    % Compute initial eigenvalue decomposition
    M = value(M);
    Z = value(Z);
    [V, D] = schur([eye(pm), M; M', Z]);
    [~, order] = sort(diag(D), 'asc');
    V_prev = V(:, order(1:mm));
    
    % Choose weighting factors
    e_prev = D(order(mm), order(mm));
    w_init = 1;
    t = 2;
    w_now = w_init * t;

end

max_iter = 1e4; % maximum number of allowed iterations
resid = zeros(1, max_iter+1); % bilinear constraint violation per iteration

% Compute post-initialization cost
resid(1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

switch method % employ selected method to solve relaxed problem

    case 1

        while iter < max_iter && resid(iter+1) > eps_safe

            % Prepare next iteration
            iter = iter + 1;
            Xka = -value(T_A);
            Yka = -value(T_B);
            d_prev = value(d);

            % Update variables
            d = d_prev;
            dbar = sdpvar(1);
            d2 = sdpvar(1);
            X = sdpvar(length(T_sys.a));
            Xbar = sdpvar(length(T_sys.a));

            % Bounded real lemma matrix
            sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + ...
                (Bf2 * Cf2)' * Xbar)', X * Bf1 + Xbar * Bf2 * ...
                Df21, (Cf1 + Df12 * Cf2 * d)'; (X * Bf1 + Xbar * ...
                Bf2 * Df21)', -eye(size(Df11'*Df11)), (Df11 + ...
                Df12 * Df21 * d)'; Cf1 + Df12 * Cf2 * d, Df11 + ...
                Df12 * Df21 * d, - eye(size(Df11*Df11'))];

            % Bilinear optimization variables
            T_C = blkdiag(Xbar, d2, d-dbar);
            T_A = blkdiag(X, dbar, 0);
            T_B = blkdiag(eye(size(X))*d, d, 0);
            [~, ma] = size(T_A);
            [pm, mm] = size(T_C);

            % Constraint for well-defined controller
            Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), (N_inf' * ...
                Y_inf), (N_inf' * N_inf)] * [d * eye(dim); dbar * ...
                eye(dim); d2 * (eye(dim))] >= eps_safe * eye(dim);

            % Constraint for implementable NRF
            Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), ...
                (Nb_inf' * Yb_inf), (Nb_inf' * Nb_inf)] * [d; dbar; d2] ...
                >= eps_safe * eye(1);

            % Constraints for norm bound
            Con_norm = sysmat <= -eps_safe * eye(size(sysmat));
            Con_X_pos = X >= eps_safe * eye(size(X));

            % Group up constraints
            Con = [Con_def, Con_implem, Con_norm, Con_X_pos];

            Obj = norm([T_C + T_A * Yka, T_A + Xka; ...
                zeros(ma, mm), eye(ma)], 'nuclear'); % relaxed objective

            sol = optimize(Con, Obj, options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end

            resid(iter+1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

            if resid(iter+1) <= eps_safe
                break; % iteration converged after odd number of steps
            end

            % Prepare next iteration
            iter = iter + 1;
            Xkb = -value(T_A);
            Ykb = -value(T_B);
            dbar_prev = value(dbar);
            X_prev = value(X);

            % Update variables
            d = sdpvar(1);
            dbar = dbar_prev;
            d2 = sdpvar(1);
            X = X_prev;
            Xbar = sdpvar(length(T_sys.a));

            % Bounded real lemma matrix
            sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + ...
                (Bf2 * Cf2)' * Xbar)', X * Bf1 + Xbar * Bf2 * Df21, ...
                (Cf1 + Df12 * Cf2 * d)'; (X * Bf1 + Xbar * Bf2 * Df21)',...
                -eye(size(Df11'*Df11)), (Df11 + Df12 * Df21 * d)'; Cf1 +...
                Df12 * Cf2 * d, Df11 + Df12 * Df21 * d, ...
                - eye(size(Df11*Df11'))];

            % Bilinear optimization variables
            T_C = blkdiag(Xbar, d2, d-dbar);
            T_A = blkdiag(X, dbar, 0);
            T_B = blkdiag(eye(size(X))*d, d, 0);
            [~, ma] = size(T_A);
            [pm, mm] = size(T_C);

            % Constraint for well-defined controller
            Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), ...
                (N_inf' * Y_inf), (N_inf' * N_inf)] * [d * ...
                eye(dim); dbar * eye(dim); d2 * (eye(dim))] >= ...
                eps_safe * eye(dim);

            % Constraint for implementable NRF
            Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), ...
                (Nb_inf' * Yb_inf), (Nb_inf' * Nb_inf)] * ...
                [d; dbar; d2] >= eps_safe * eye(1);

            % Constraints for norm bound
            Con_norm = sysmat <= -eps_safe * eye(size(sysmat)); %X is fixed

            % Group up constraints
            Con = [Con_def, Con_implem, Con_norm];

            Obj = norm([T_C + Xkb * T_B, zeros(pm, ma); ...
                T_B + Ykb, eye(ma)], 'nuclear'); % nuclear norm relaxation

            sol = optimize(Con, Obj, options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end

            resid(iter+1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

        end

    case 2

        while iter < max_iter && resid(iter+1) > eps_safe

            % Prepare next iteration
            iter = iter + 1;
            Xk = -blkdiag(value(X), value(dbar), 0);
            Yk = -blkdiag(eye(size(X))*value(d), value(d), 0);

            % Update variables
            d = sdpvar(1);
            dbar = sdpvar(1);
            d2 = sdpvar(1);
            X = sdpvar(length(T_sys.a));
            Xbar = sdpvar(length(T_sys.a));
            sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + ...
                (Bf2 * Cf2)' * Xbar)', X * Bf1 + Xbar * Bf2 * ...
                Df21, (Cf1 + Df12 * Cf2 * d)'; (X * Bf1 + Xbar * ...
                Bf2 * Df21)', -eye(size(Df11'*Df11)), (Df11 + ...
                Df12 * Df21 * d)'; Cf1 + Df12 * Cf2 * d, Df11 + ...
                Df12 * Df21 * d, -eye(size(Df11*Df11'))];

            % Bilinear optimization variables
            T_C = blkdiag(Xbar, d2, d-dbar);
            T_A = blkdiag(X, dbar, 0);
            T_B = blkdiag(eye(size(X))*d, d, 0);
            [~, ma] = size(T_A);
            [pm, mm] = size([T_C, T_A; T_B, eye(ma)]);

            % Constraint for well-defined controller
            Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), ...
                (N_inf' * Y_inf), (N_inf' * N_inf)] * [d * ...
                eye(dim); dbar * eye(dim); d2 * (eye(dim))] >= ...
                1e-6 * eye(dim);

            % Constraint for implementable NRF
            Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), ...
                (Nb_inf' * Yb_inf), (Nb_inf' * Nb_inf)] * ...
                [d; dbar; d2] >= 1e-6 * eye(1);

            % Constraints for norm bound
            Con_norm = sysmat <= -1e-6 * eye(size(sysmat));
            Con_X_pos = X >= 1e-6 * eye(size(X));

            % Group up constraints
            Con = [Con_def, Con_implem, Con_norm, Con_X_pos];

            Obj = norm([T_C + Xk * Yk + T_A * Yk + Xk * T_B, T_A + Xk; ...
                T_B + Yk, eye(ma)], 'nuclear'); % relaxed objective

            sol = optimize(Con, Obj, options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end

            resid(iter+1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

        end

    case 3

        rho_2 = 1e4; % choose the two step sizes
        rho_1 = 5 * rho_2; % ensure that rho_1 > 4 * rho_2 to guarantee
        % method convergence

        Lam_f = rho_1 * [value(T_C) - value(T_A) * value(T_B), ...
            zeros(size(value(T_A))); zeros(size(value(T_B))), ...
            zeros(ma)];
        M_bar_f = [value(T_C), value(T_A); value(T_B), eye(ma)];
        M_f = [value(T_C), value(T_A); value(T_B), eye(ma)];
        MS_bar_f = [value(T_A); eye(ma)];
        MD_bar_f = [value(T_B), eye(ma)];

        time_limit = 900; % impose timeout limit
        tic

        while iter < max_iter && resid(iter+1) > eps_safe

            % Prepare next iteration
            iter = iter + 1;

            % Update variables
            d = sdpvar(1);
            dbar = sdpvar(1);
            d2 = sdpvar(1);
            X = sdpvar(length(T_sys.a));
            Xbar = sdpvar(length(T_sys.a));
            sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + ...
                (Bf2 * Cf2)' * Xbar)', X * Bf1 + Xbar * Bf2 * ...
                Df21, (Cf1 + Df12 * Cf2 * d)'; (X * Bf1 + Xbar * ...
                Bf2 * Df21)', -eye(size(Df11'*Df11)), (Df11 + ...
                Df12 * Df21 * d)'; Cf1 + Df12 * Cf2 * d, Df11 + ...
                Df12 * Df21 * d, -eye(size(Df11*Df11'))];

            % Bilinear optimization variables
            T_C = blkdiag(Xbar, d2, d-dbar);
            T_A = blkdiag(X, dbar, 0);
            T_B = blkdiag(eye(size(X))*d, d, 0);
            [~, ma] = size(T_A);
            [px, mx] = size([T_A; eye(ma)]);
            MS_bar = sdpvar(px, mx, 'full');
            [px, mx] = size([T_B, eye(ma)]);
            MD_bar = sdpvar(px, mx, 'full');
            M = [T_C, T_A; T_B, eye(ma)];
            [pm, mm] = size(M);
            M_bar = sdpvar(pm, mm, 'full');

            % Auxiliary variables
            x1 = sdpvar(1);
            x2 = sdpvar(1);
            x3 = sdpvar(1);
            x4 = sdpvar(1);
            x5 = sdpvar(1);
            x6 = sdpvar(1);

            % Constraint for well-defined controller
            Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), ...
                (N_inf' * Y_inf), (N_inf' * N_inf)] * [d * ...
                eye(dim); dbar * eye(dim); d2 * (eye(dim))] >= ...
                1e-6 * eye(dim);

            % Constraint for implementable NRF
            Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), ...
                (Nb_inf' * Yb_inf), (Nb_inf' * Nb_inf)] * ...
                [d; dbar; d2] >= 1e-6 * eye(1);

            % Constraints for norm bound
            Con_norm = sysmat <= -1e-6 * eye(size(sysmat));
            Con_X_pos = X >= 1e-6 * eye(size(X));

            % Constraints based upon auxiliary variables
            cone_1 = cone(reshape(M-M_bar_f, pm*mm, 1), x1);
            cone_2 = cone(reshape(M_bar_f-MS_bar*MD_bar_f, pm*mm, 1), x2);
            trace_con_1 = trace(Lam_f*(M_bar_f - MS_bar * MD_bar_f)') <=x3;
            pos_def_cons_1 = [x1 >= 0, x2 >= 0, x3 >= 0];

            % Group up constraints
            Con = [Con_def, Con_implem, Con_norm, Con_X_pos ...
                cone_1 cone_2 trace_con_1 pos_def_cons_1];

            % Use auxiliary variables to form objective
            Obj = (rho_2 / 2) * x1 + x3 + (rho_1 / 2) * x2;

            sol = optimize(Con, Obj, options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end

            % Set up subsequent optimization step
            M_f = [value(T_C), value(T_A); value(T_B), eye(ma)];
            MS_bar_f = value(MS_bar);

            % Constraints based upon auxiliary variables
            cone_3 = cone(reshape(M_f-M_bar, pm*mm, 1), x4);
            cone_4 = cone(reshape(M_bar-MS_bar_f*MD_bar, pm*mm, 1), x5);
            trace_con_2 = trace(Lam_f*(M_bar - MS_bar_f * MD_bar)') <= x6;
            pos_def_cons_2 = [x4 >= 0, x5 >= 0, x6 >= 0];
            Con = [cone_3 cone_4 trace_con_2 pos_def_cons_2];

            % Use auxiliary variables to form objective
            Obj = (rho_2 / 2) * x4 + x6 + (rho_1 / 2) * x5;

            sol = optimize(Con, Obj, options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end

            % Set up initial optimization step for next iteration
            M_bar_f = value(M_bar);
            MD_bar_f = value(MD_bar);
            Lam_f = Lam_f + rho_1 * (M_bar_f - MS_bar_f * MD_bar_f);

            resid(iter+1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

            % Check if method has timed out
            loop_time = toc;
            if loop_time >= time_limit
                break
            end

        end

    case 4

        time_limit = 900; % impose timeout limit
        tic

        while iter < max_iter && resid(iter+1) > eps_safe

            % Prepare next iteration
            iter = iter + 1;

            % Update variables
            d = sdpvar(1);
            dbar = sdpvar(1);
            d2 = sdpvar(1);
            X = sdpvar(length(T_sys.a));
            Xbar = sdpvar(length(T_sys.a));
            sysmat = [(Af' * X + (Bf2 * Cf2)' * Xbar) + (Af' * X + ...
                (Bf2 * Cf2)' * Xbar)', X * Bf1 + Xbar * Bf2 * ...
                Df21, (Cf1 + Df12 * Cf2 * d)'; (X * Bf1 + Xbar * ...
                Bf2 * Df21)', -eye(size(Df11'*Df11)), (Df11 + ...
                Df12 * Df21 * d)'; Cf1 + Df12 * Cf2 * d, Df11 + ...
                Df12 * Df21 * d, -eye(size(Df11*Df11'))];

            % Bilinear optimization variables
            T_C = blkdiag(Xbar, d2, d-dbar);
            T_A = blkdiag(X, dbar, 0);
            T_B = blkdiag(eye(size(X))*d, d, 0);
            [~, ma] = size(T_A);
            M = [T_C, T_A; T_B, eye(ma)];
            [pm, mm] = size(M);
            Y = sdpvar(pm);
            Z = sdpvar(mm);
            e_now = sdpvar(1);

            % Constraint for well-defined controller
            Con_def = Y_inf' * Y_inf + [(Y_inf' * N_inf), ...
                (N_inf' * Y_inf), (N_inf' * N_inf)] * [d * ...
                eye(dim); dbar * eye(dim); d2 * (eye(dim))] >= ...
                1e-6 * eye(dim);

            % Constraint for implementable NRF
            Con_implem = Yb_inf' * Yb_inf + [(Yb_inf' * Nb_inf), ...
                (Nb_inf' * Yb_inf), (Nb_inf' * Nb_inf)] * ...
                [d; dbar; d2] >= 1e-6 * eye(1);

            % Constraints for norm bound
            Con_norm = sysmat <= -1e-6 * eye(size(sysmat));
            Con_X_pos = X >= 1e-6 * eye(size(X));

            % Constraints based upon auxiliary variables
            pos_sem_def_1 = [Y, M; M', Z] >= 0 * eye(pm+mm);
            pos_sem_def_2 = [eye(pm), M; M', Z] >= 0 * eye(pm+mm);
            pos_sem_def_3 = e_now * eye(pm) - V_prev' * [eye(pm), M; ...
                M', Z] * V_prev >= 0 * eye(pm);
            e_ineq = e_prev - e_now >= 0;

            % Group up constraints
            Con = [Con_def, Con_implem, Con_norm, Con_X_pos ...
                pos_sem_def_1 pos_sem_def_2 pos_sem_def_3 e_ineq];

            % Define an objective
            Obj = trace(Y) + w_now * e_now;

            sol = optimize(Con,Obj,options);

            % Analyze error flags
            if sol.problem ~= 0
                disp('Hmm, something went wrong!');
                sol.info
                yalmiperror(sol.problem)
            end
            
            % Compute iteration's eigenvalue decomposition
            M = value(M);
            Z = value(Z);
            [V, D] = schur([eye(pm), M; M', Z]);
            [~, order] = sort(diag(D), 'asc');

            % Update weighting factors
            e_prev = e_now;
            w_now = w_now * t;

            resid(iter+1) = sum(svd(value(T_C)-value(T_A)*value(T_B)));

            % Check if method has timed out
            loop_time = toc;
            if loop_time >= time_limit
                break
            end

        end

    otherwise

        error('Method selection invalid!');

end

run_time = toc; % time elapsed during optimization

%%
%  Obtain the Youla parameter which ensures NRF structure and robustness

Qs = tf(value(d), 1) * eye(dim); % express the solution of the optimization
Q = (Tk \ Qs * eye(dim)); % form the Youla parameter for the original DCF

K_LF = balreal(ss(Tk \ (Ytb * Tk) + Q * Ntb * Tk, 'min')); % left factor
K_RF = balreal(ss(Tk \ (Xtb * eye(dim)) + Q * Mtb, 'min')); % right factor
K = balreal(ss(K_LF \ K_RF, 'min')); % the controller's structureless TFM

%%
%  Form closed-loop system

T1e = dss([Ar, -Br * Fa; zeros(length(Aa), length(Ar)), Aa + La * Ca], ...
    [zeros(length(Ar), dim), -Br; La, -Ba - La * Da], eps_m* ...
    [-Hr * Fr, -Hr * Fa], [zeros(dim), -eps_m * Hr], ...
    blkdiag(eye(size(Ar)), Ea));
T1e = balreal(ss(T1e, 'min')); % eliminate nondynamic modes


T2e = ss(Ar, Br, - eps_m * Hr * Fr, eps_m * Hr); % no nondynamic modes
T2e = balreal(ss(T2e, 'min'));

T3e = dss(Aa + La * Ca, [La, - Ba - La * Da], Ca, [eye(dim), - Da], Ea);
T3e = balreal(ss(T3e, 'min')); % eliminate nondynamic modes

T_CL_r = T1e + T2e * Q * T3e;
T_CL_r = balreal(ss(T_CL_r, 'min'));

rob_marg_1 = eps_m / norm(T_CL_r, inf); % greater than eps_m
disp(rob_marg_1) % display obtained stability radius for approximation

rob_marg_2 = ncfmargin(G, K, 1, 1e-6); % greater than dist_sup
disp(rob_marg_2) % validate previous computation

rob_marg_3 = ncfmargin(G_bar, K, 1, 1e-6); % good robustness indicator
disp(rob_marg_3) % display obtained stability radius for original network

%%
%  Form controller NRF

K_LF_struc = Tk * K_LF; % form structured left  factor of controller LCF
K_RF_struc = Tk * K_RF; % form structured right factor of controller LCF

Phi = tf(0, 1) * ones(dim);
Gamma = tf(0, 1) * ones(dim);
Phi(1, dim) = -balreal(ss(K_LF_struc(1, 1) \ K_LF_struc(1, dim), 'min'));
Gamma(1, 1) = balreal(ss(K_LF_struc(1, 1) \ K_RF_struc(1, 1), 'min'));

for i = 2:dim
    Phi(i, i - 1) = -balreal(ss(K_LF_struc(i, i) \ K_LF_struc(i, i - 1),...
        'min'));
    Gamma(i, i) = balreal(ss(K_LF_struc(i, i) \ K_RF_struc(i, i), 'min'));
end

%  Notice that Phi_K is identical, up to numerical tolerance, with its
%  feedthrough term and preserve only the latter

Phi = ss(Phi); % obtain realization
norm(Phi - Phi.d, inf) % check against feedthrough term
% approximately 0
Phi = Phi.d; % preserve only feedthrough

norm(K - (eye(dim) - Phi) \ Gamma, inf) % check NRF validity
% approximately 0

%%
%  Approximate solution

Phi_a = Phi; % preserve feedforward term
Gamma_a = floor(evalfr(Gamma, inf)); % approximate feedback term
K_a = (eye(dim) - Phi_a) \ Gamma_a; % form apprixmated controller's TFM

rob_marg_a = ncfmargin(G_bar, K_a, 1, 1e-6);% a good robustness indicator
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
