clear all; clc

%% Benchmark Parameters
tau_c=0.075;
tau_y=0.30;
beta=0.96;
sigma=0.2;
gamma=0.4;
alpha=0.4;
delta=0.05;
rho=0.9;
sigma_e=0.2;

%% Get theta grid
% set number of theta grids
n_theta=7;
% define bounds of theta grid in terms of standard deviations
m=1.645;    %will update probably, but 1.645 is roughly 90%, 196 is 95%                                      

[logtheta, Pr_theta]=tauchen(n_theta, rho, sigma_e, m);

%% Create Next period grids

% start with next period investment
% number of grid spaces

n_a=350; % will adjust this later

ub_a=10; % upper bound is placeholder

a_grid=linspace(0, ub_a, n_a); % lower bound is 0 by condition of model
% labour grid
n_l=100;
l_grid=linspace(0.01,0.99, n_l); % end at 0.99 to avoid dividing by zero

%% Initial rental rate and wage
% this is a guess
r=(1/beta-1)*(1-tau_y);
% given by FOCs
w=(1-alpha)*((delta+r)/alpha)^(alpha/(1-alpha));

%% Test EGM

[a,c,l,v]=EGM(tau_c, tau_y, beta, sigma, gamma, logtheta, Pr_theta, a_grid, l_grid, r, w, 1e-4);


%% tauchen method;
function [theta, Pr_theta] = tauchen(n, p, sigma, m)
   z=zeros(n,1);
   z_p=zeros(n,n);
   z(n)=m*sigma/sqrt(1-p^2);
   z(1)=-z(n);
   step=(z(n)-z(1))/(n-1);
   for i = 2:(n-1)
        z(i) = z(1) + (i-1)*step;
   end
   z_p(:,1)=cdf("normal", (z(1)-p*z+step/2), 0, sigma);
   for i=2:(n-1)
       z_p(:,i)=cdf("normal", (z(i)-p*z+step/2), 0, sigma)-cdf("normal", (z(i)-p*z-step/2), 0, sigma);
   end
   z_p(:,n)=1-cdf("normal", z(n)-p*z-step/2, 0, sigma);
   theta=z;
   Pr_theta=z_p;

end



%% EGM

function [a_star, c_star, l_star, v_star]= EGM(tc, ty, b, s, g, thetas, p_thetas, a_grid, l_grid, r, w, tol)
    %extract grid lengths
    n_a=length(a_grid);
    n_t=length(thetas);
    %create initial guesses
    x_a=ones(n_a, n_t);     %start with consumption being positive
    n=0.0*ones(n_a, n_t);   % and labour being weakly positive
    g_a=zeros(n_a, n_t);    % this is just a place holder, guess of capital is given by a_grid
    % initialize distance
    dist_c=Inf;
    % Define Marginal Utility function wrt consumption
    MU = @(c,l) c^((1-s)*g-1)*g*(1-l)^((1-s)*(1-g));
    % Define the Utility Function
    U = @(c,l) ((c^g*(1-l)^(1-g))^(1-s))/(1-s);
    % define consumption from budget constraint
    BC = @(l, a, ap, t) ((1-ty)*w*t*l+(1+r*(1-ty))*a-ap)/(1+tc);
    % save this here so that it's not calculated hundreds of times
    exp_c=(1-s)*g-1;
    i=0;
    while dist_c>tol
        i=i+1 % keep track of iterations
        % create updated guess
        Tx=zeros(n_a, n_t);
        Tn=zeros(n_a, n_t);
        Tg=zeros(n_a, n_t);
        % Want to store value function
        Tv=zeros(n_a, n_t);
        % create expected value next period
        for i_t=1:n_t
            t=exp(thetas(i_t)); %extract value of theta
            MU_p=zeros(n_a, 1); %store marginal utilities
            U_p=zeros(n_a,1);   % store utilities
            c_euler=zeros(n_a, 1);   %store consumption
            l_euler=zeros(n_a, 1);   % labour
            a_euler=zeros(n_a, 1);   % and investment
            for i_a=1:n_a   %this loop is just to get RHS of Euler equation
                E_MU=0;  % store expected RHS
                EU_p=0;  % store expected value function
                for j_t=1:n_t  %iterate over stochastic transitions
                    %c_p=x_a(i_a, i_t);  %assumed level of consumption
                    l_p=n(i_a, j_t);    %assumed level of labour
                    a_p=a_grid(i_a);     %assumed level of wealth in this period

                    a_pp=g_a(i_a, j_t);  %assumed level of future investment
                    %back out consumption from budget constraint
                    c_guess= BC(l_p, a_p, a_pp, exp(thetas(j_t))); %this should be weakly positive
                    %if not
                    c_p=max(c_guess, 0.0001); % consumption needs to be strictly positive

                    %c_p=x_a(i_a, j_t);
                    % add marginal utility to expected discounted by
                    % transition probability
                    E_MU=E_MU+p_thetas(i_t, j_t)*MU(c_p,l_p);
                    EU_p=EU_p+p_thetas(i_t, j_t)*U(c_p,l_p);
                end %transitions
                MU_p(i_a)=E_MU;  % store MU part of RHS
                U_p(i_a)=EU_p;
            end
            for i_a=1:n_a   % iterate over choice grid
                RHS=b*(1+r*(1-ty))*MU_p(i_a);   %RHS of Euler Equation
                RHS_U=b*(1+r*(1-ty))*U_p(i_a);  %Storing this to recover Value function
                % solve FOCs
                l_guess=n(i_a, i_t);
                dist_l=Inf; %want to solve for fixed point
                j=0;%        want to break loop in testing
                lambda=0.3; %smoothing parameter
                while dist_l>tol
                    coef_c=g*(1-l_guess)^((1-s)*(1-g));
                    c_euler(i_a)=(RHS/coef_c)^(1/exp_c);
                    l_euler(i_a)=1-(1-g)/g*((1+tc)*c_euler(i_a)/((1-ty)*w*t));
                    %clamp l to grid
                    %l_euler(i_a)=max(min(l_euler(i_a), 0.99), 0.01);
                    dist_l=abs(l_guess-l_euler(i_a));
                    l_guess=lambda*l_guess+(1-lambda)*l_euler(i_a);
                    j=j+1;
                    if j>100
                        break
                    end % max iterations
                end % intratemporal optimization
                % back out wealth from budget constaint
                a_euler(i_a)=((1+tc)*c_euler(i_a)+a_grid(i_a)-(1-ty)*w*t*l_euler(i_a))/(1+r*(1-ty));
            end % finishing making functions for current wealth
            for i_a=1:n_a   % need to interpolate
                a=a_grid(i_a);
                c_opt=max(interp1(a_euler, c_euler, a, 'linear', 'extrap'), 1e-8);% consumption must be positive;
                l_opt=max(min(interp1(a_euler, l_euler, a, 'linear', 'extrap'), 0.9999), 0.0);
                a_opt=max(min(interp1(a_euler, a_grid, a, 'linear', 'extrap'), a_grid(n_a)), a_grid(1));
                % Compute value function
                V_opt=U(c_opt, l_opt)+RHS_U;

                % store Coleman operator
                Tx(i_a, i_t)=c_opt;
                Tv(i_a, i_t)=V_opt;
                Tg(i_a, i_t)=a_opt;
                Tn(i_a, i_t)=l_opt;
            end     %over investment
        end %over stochastic state space
        
        dist_c=norm(x_a-Tx, Inf);
        x_a=Tx;
        n=Tn;
        %g_a=Tg;
        if i>500
            break
        end
    end %convergence
    c_star=Tx;
    a_star=Tg;
    l_star=Tn;
    v_star=Tv;
end
              



            
    








