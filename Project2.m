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
% define upper bound of theta grid
m=1.2;

[logtheta, Pr_theta]=tauchen(n_theta, rho, sigma_e, m);

%% Create Next period grids

% start with next period investment
% number of grid spaces

n_a=100; % will adjust this later

ub_a=20; % upper bound is placeholder

a_grid=linspace(0, ub_a, n_a); % lower bound is 0 by condition of model
% labour grid
n_l=20;
l_grid=linspace(0.01,0.99, n_l); % end at 0.99 to avoid dividing by zero

%% Initial rental rate and wage
% this is a guess
r=0.1;
% given by FOCs
w=(1-alpha)*((delta+r)/alpha)^(alpha/(1-alpha));

%% Test EGM

[a,c,l]=EGM(tau_c, tau_y, beta, sigma, gamma, logtheta, Pr_theta, a_grid, l_grid, r, w, 1e-4);


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

function [a_star, c_star, l_star]= EGM(tc, ty, b, s, g, thetas, p_thetas, a_grid, l_grid,  r, w, tol)
    % get grid lengths
    n_a=length(a_grid); % investment
    n_theta= length(thetas); % random space
    %initialize grids
    g_a=zeros(n_a, n_theta); % Initial guess of investment
    x_a=ones(n_a, n_theta)*0.5; % initial guess for consumption
    n=0.3*ones(n_a, n_theta); % initial guess of labour
    V=ones(n_a, n_theta)*(-Inf);  % Value function
    % initialize distances
    dist_x=Inf;
    dist_g=Inf;
    dist_n=Inf;
    %start iteration
    i=0;
    MU = @(c,l) c^((1-s)*g-1)*g*(1-l)^((1-s)*(1-g));
    while dist_x>tol
        %initialize bellman/coleman operator
        Tg_a=zeros(n_a, n_theta);
        Tx_a=zeros(n_a, n_theta);
        Tn=zeros(n_a, n_theta);
        TV=zeros(n_a, n_theta);
        i=i+1% want to track number of iterations
        for i_t=1:n_theta %iterate over shocks
            t=exp(thetas(i_t));
            %initialize RHS of Euler equation
            MU_p=zeros(n_a, 1);
            for i_a=1:n_a; % iterate over capital choice next period
                % iterate over expected RHS of Euler
                E_MU=0;
                for j_t=1:n_theta
                    c_p=x_a(i_a, j_t);  % assumed capital in RHS of Euler
                    l_p=n(i_a, j_t);    % assumed labour in RHS of Euler
                    MU_j= MU(c_p,l_p) ;  % compute the value of one realization
                    E_MU=E_MU+p_thetas(i_t, j_t)*MU_j;  % add said value to expectation discounted by probability
                end     % expected value of marginal utility
                MU_p(i_a)=E_MU; % store value in grid
            end %over capital decision
            for i_a=1:n_a
                a_p=a_grid(i_a);
                U_opt=-Inf; % store current period utility
                c_opt=1;    %consumption
                l_opt=0.5;    %labour
                a_opt=0;    % and investment
                RHS=b*(1+r*(1-ty))*MU_p(i_a);
                %for l=l_grid;
                l=n(i_a, i_t);

                    % Solve for C from the Euler Equation
                    exponent_c= (1-s)*g-1; % The exponent on c from the marginal utility
                    coef = g*(1-l)^((1-s)*(1-g));    %the coefficient on c from MU
                    c = (RHS/coef)^(1/exponent_c);
                    l=1-((1-g)/g)*(1+tc)*c/((1-ty)*w*t);
                    l=max(min(l, 0.99), 0.01);
                    coef = g*(1-l)^((1-s)*(1-g));
                    c = (RHS/coef)^(1/exponent_c);
                    l_opt=l;
                    %check feasibility
                    if c >0 && c<Inf %assume holds striclty becuase Inada conditions
                        % compute a

                        a = (a_p+(1+tc)*c-(1-ty)*w*t*l)/(1+r*(1-ty));
                        % check feasibility
                        if a >=0 
                            U=((c^g*(1-l)^(1-g))^(1-s))/(1-s); %compute the utility function
                            if U>U_opt % check optimality
                                %store optimal values
                                U_opt=U;
                                c_opt=c;
                                l_opt=l;
                                a_opt=a;
                            end% no need for else here
                        else
                            continue % go to next l if investment infeasible
                        end
                    else
                        continue % go to next l if consumption infeasible
                    end
                %end % l_grid
                Tx_a(i_a, i_t)=c_opt;
                Tg_a(i_a, i_t)=a_opt;
                Tn(i_a, i_t)= l_opt;
                TV(i_a, i_t)= U_opt+RHS;
            end % a_grid
        end %thetas
    % update fixed point math
    dist_x=norm(x_a-Tx_a, Inf);
    dist_g=norm(g_a-Tg_a, Inf);
    dist_n=norm(n-Tn,Inf);

    x_a=Tx_a;
    g_a=Tg_a;
    n=Tn;
           

    if i>1000
        break
    end

                    
    end % convergence
    a_star=g_a;
    c_star=x_a;
    l_star=n;
end








