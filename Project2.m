clear all; clc

%% Benchmark Parameters
tau_c=0.075;
tau_y=0.30;
beta=0.96;
sigma=2.0;
gamma=0.4;
alpha=0.4;
delta=0.05;
rho=0.9;
sigma_e=0.2;

%% Get theta grid
% set number of theta grids
n_theta=7;
% define bounds of theta grid in terms of standard deviations
cv=0.01;
m=icdf('normal', 1-cv/2, 0, 1);   %will update probably, but 1.645 is roughly 90%, 1.96 is 95%                                      

[logtheta, Pr_theta]=tauchen(n_theta, rho, sigma_e, m);

%% Create Next period grids

% start with next period investment
% number of grid spaces



ub_a=exp(m)*2; % upper bound is placeholder
n_a=500; % will adjust this later 
%non-linear grid because density is highest in the poorer regions
a_grid=linspace(0, sqrt(ub_a), n_a).^2; % lower bound is 0 by condition of model

%% Initial rental rate and wage
% this is a guess
%r=0.10;
% given by FOCs
%w=(1-alpha)*((delta+r)/alpha)^(alpha/(1-alpha));

%% Test EGM

%[a,c,l]=EGM(tau_c, tau_y, beta, sigma, gamma, logtheta, Pr_theta, a_grid, r, w, 1e-4);

%% Test Stationary Distribution

%p=distribution(a_grid, a, Pr_theta, 1e-4);

%% Test Market Clearing

%[K,L,C,G]=aggregates(a, l, c, p, logtheta, tau_y, tau_c, delta, alpha);

%r_market=alpha*(K/L)^(alpha-1)-delta;
%w_market=(1-alpha)*(K/L)^alpha;

%% Solve Benchmark Model
[a_b,c_b,l_b, G_b, r_b, w_b, psi_b, Y_b, C_b] = solve_model(a_grid,logtheta, Pr_theta, tau_c, tau_y, delta, alpha, beta, gamma, sigma);
v_b=recover_value(a_grid, a_b, c_b, l_b, beta, gamma, sigma, Pr_theta);

%% Solve Proposal Economy
[a_r,c_r,l_r, r_r, w_r, tau_r, psi_r, Y_r, C_r] = solve_reform(a_grid,logtheta, Pr_theta, delta, alpha, beta, gamma, sigma,G_b);
v_r=recover_value(a_grid, a_r, c_r, l_r, beta, gamma, sigma, Pr_theta);

%% Compute Welfare Gains

ce1=CE1(v_b, v_r, gamma, sigma);
cbar1=pseudo_int(ce1, psi_b);

ce2=CE2(v_b, v_r, psi_b, psi_r, gamma, sigma);

%% Save Results
Vb=sum(sum(v_b.*psi_b));
Vr=sum(sum(v_r.*psi_r));


aggs=[Vb C_b Y_b; Vr, C_r, Y_r; Vr-Vb, C_r-C_b,Y_r-Y_b];
agg_row={'Benchmark', 'Reform', 'Change'};
agg_column={'V', 'C', 'Y'};
%running this will make the table ugly
matrix2latex(aggs, 'aggs.tex', 'rowLabels', agg_row, 'columnLabels', agg_column, 'format', '%.4f');

%% More Results

ce1_row={'90+', '80-90', '60-80', '40-60', '20-40', '0-20'};
ce1_col={1,2,3,4,5,6,7};
matrix2latex(cbar1, 'cbar.tex','rowLabels', ce1_row, 'columnLabels', ce1_col, 'format', '%.4f')


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

function [a_star, c_star, l_star]= EGM(tc, ty, b, s, g, thetas, p_thetas, a_grid, r, w, tol)
    %extract grid lengths
    n_a=length(a_grid);
    n_t=length(thetas);
    %create initial guesses
    x_a=zeros(n_a, n_t);     %Will fill this shortly
    n=0.3*ones(n_a, n_t);   % and labour being weakly positive
    g_a=zeros(n_a, n_t);    % this is just a place holder, guess of capital is given by a_grid
    % initialize distance
    dist_c=Inf;
    % Define Marginal Utility function wrt consumption
    MU = @(c,l) c^((1-s)*g-1)*g*(1-l)^((1-s)*(1-g));
    % define consumption from budget constraint
    BC = @(l, a, ap, t) ((1-ty)*w*t*l+(1+r*(1-ty))*a-ap)/(1+tc);
    % save this here so that it's not calculated hundreds of times
    exp_c=(1-s)*g-1;
    lambda_l=0.3; %smoothing parameter
    lambda_t=0.5;
    for i=1:n_a %set initial consumption guess to be coh
        for j=1:n_t
            x_a(i,j)=BC(n(i,j), a_grid(i), g_a(i,j), exp(thetas(j)));
        end
    end
    i=0;
    while dist_c>tol
        i=i+1; % keep track of iterations
        % create updated guess
        Tx=zeros(n_a, n_t);
        Tn=zeros(n_a, n_t);
        Tg=zeros(n_a, n_t);
        % create expected value next period
        for i_t=1:n_t
            t=exp(thetas(i_t)); %extract value of theta
            MU_p=zeros(n_a, 1); %store marginal utilities
            c_euler=zeros(n_a, 1);   %store consumption
            l_euler=zeros(n_a, 1);   % labour
            a_euler=zeros(n_a, 1);   % and investment
            for i_a=1:n_a   %this loop is just to get RHS of Euler equation
                E_MU=0;  % store expected RHS
                for j_t=1:n_t  %iterate over stochastic transitions
                    c_p=x_a(i_a,j_t);   %assumed level of consumption
                    l_p=n(i_a, j_t);    %assumed level of labour
                    % add marginal utility to expected discounted by
                    % transition probability
                    E_MU=E_MU+p_thetas(i_t, j_t)*MU(c_p,l_p);
                end %transitions
                MU_p(i_a)=E_MU;  % store MU part of RHS
            end
            for i_a=1:n_a   % iterate over choice grid
                RHS=b*(1+r*(1-ty))/(1+tc)*MU_p(i_a);   %RHS of Euler Equation
                % solve FOCs
                l_guess=n(i_a, i_t);
                dist_l=Inf; %want to solve for fixed point
                j=0;%        want to break loop in testing
                
                while dist_l>tol
                    % this is defined on each iteration of the loop since we need a guess of labour
                    coef_c=g*(1-l_guess)^((1-s)*(1-g)); 
                    c_euler(i_a)=(RHS/coef_c)^(1/exp_c);
                    l_euler(i_a)=1-(1-g)/g*((1+tc)*c_euler(i_a)/((1-ty)*w*t));
                    %clamp l to grid
                    l_euler(i_a)=max(min(l_euler(i_a), 0.9999), 0.0);
                    dist_l=abs(l_guess-l_euler(i_a));
                    l_guess=lambda_l*l_guess+(1-lambda_l)*l_euler(i_a);
                    j=j+1;
                    if j>80 % don't need that many iterations since we still have outer loop
                        break
                    end % max iterations
                end % intratemporal optimization
                % back out wealth from budget constaint
                a_euler(i_a)=((1+tc)*c_euler(i_a)+a_grid(i_a)-(1-ty)*w*t*l_guess)/(1+r*(1-ty));
              
            end % finishing making functions for current wealth
            for i_a=1:n_a   % need to interpolate
                a=a_grid(i_a);
                a_opt=max(min(interp1(a_euler, a_grid, a, 'linear', 'extrap'), a_grid(n_a)), a_grid(1));    %clamp to grid
                %c_opt=max(interp1(a_euler, c_euler, a, 'linear', 'extrap'), 1e-8);% consumption must be positive;
                %l_opt=max(min(interp1(a_euler, l_euler, a_opt, 'linear', 'extrap'), 0.9999), 0.0);  %clamp to grid
                %a_opt=max(min((((1+tc)*c_opt+a-(1-ty)*w*t*l_opt))/(1+r*(1-ty)),a_grid(n_a)),a_grid(1)) ;
                c_opt=max(BC(l_euler(i_a), a_grid(i_a), a_opt, t),1e-8);       
                % store Coleman operator
                Tx(i_a, i_t)=c_opt;
                Tg(i_a, i_t)=a_opt;
                Tn(i_a, i_t)=l_euler(i_a);
            end     %over investment
        end %over stochastic state space
        
        dist_c=norm(x_a-Tx, Inf);
        x_a=lambda_t*x_a+(1-lambda_t)*Tx;
        g_a=lambda_t*g_a+(1-lambda_t)*Tg;
        n=lambda_t*n+(1-lambda_t)*Tn;
        if i>1000
            break
        end
    end %convergence
    c_star=Tx;
    a_star=Tg;
    l_star=Tn;
end

%% Stationary Distribution

function [v] = distribution(a_p, a, transitions, tol)% takes a_grid, a_star, and Pr_theta
    
    %extract grid lengths
    [n_a, n_t]=size(a);
    % need to find where wealth points to in transitions
    Idx_at=zeros(n_a, n_t); % matrix of index pointers
    Wgt_at=zeros(n_a, n_t); % matrix of weights to divide wealth
    for ia=1:n_a    %iterate over investment grid
        for it=1:n_t    % iterate over shocks
            g = a(ia, it); % This investment policy of person at wealth ia at shock it
            if g<a_p(1)     % if lower than lower bound of grid
                Idx_at(ia, it)=1;   %point to lower bound of grid
                Wgt_at(ia,it)=1;     % assign full weight
            elseif g> a_p(n_a)      %if above upper bound
                Idx_at(ia,it)=n_a;   % point to upper bound
                Wgt_at(ia,it)=1;     %assign full weight
            else
                i=1;    %start count (already checked 1)
                while a_p(i+1)<g    % no need for bounds since they are the other part of if
                    i=i+1;  %keep looking
                end
            end % at this point we found the infemum
            Idx_at(ia,it)=i;    % point at infemum
            ap_i=a_p(i);
            ap_i2=a_p(i+1);
            Wgt_at(ia, it)=(ap_i2-g)/(ap_i2-ap_i);
        end % shocks
    end     % grid
    %at this point we have matrix of transition locations and weights
    psi=ones(n_a,n_t)/(n_a*n_t);  % initial guess is uniform distribution

    dist_p=Inf;
    i=0;    % keep track of number of iterations
    while dist_p>tol   %find fixed point
        i=i+1;
        Tpsi=zeros(n_a, n_t);   %next guess
        for ia=1:n_a    %iterate over wealth
            for it=1:n_t    %iterate over shocks
                to=Idx_at(ia,it);        % want where this person goes
                wgt=Wgt_at(ia,it);      % want the weight
                for jt=1:n_t    % iterate over path
                    Tpsi(to, jt)= Tpsi(to,jt) + ...    %add to what we have
                        wgt*...                        % the weight
                        psi(ia,it)*...                 % the mass of being in ia it
                        transitions(it,jt);            % probability of going from it to jt
                    Tpsi(to+1, jt) = Tpsi(to+1, jt) + ...
                        (1-wgt)*...
                        psi(ia, it)* ...
                        transitions(it, jt);
                end
            end
        end% done updating
        dist_p=norm(Tpsi-psi, Inf);
        psi=Tpsi;
    end % have convergence
    v=psi;
end

%% Get Aggregates

function [K,L,C,G, Y] = aggregates(a, n, x, psi, theta, ty, tc,d,alpha)
    %get dimensions
    [na,nt]=size(a);
    t=exp(theta);
    k=zeros(1, nt);
    l=zeros(1,nt);
    c=zeros(1,nt);
    for i=1:na  %sum across all investment decisions
            k=k+sum(a(i,:).*psi(i,:));
            l=l+sum(t.*n(i,:).*psi(i,:));
            c=c+sum(x(i,:).*psi(i,:));
    end
    K=sum(k);
    L=sum(l);
    C=sum(c);
    Y=K^alpha*L^(1-alpha);
    G=ty*(Y-K*d)+tc*C;
end

%% Recover Value Function
function [V]= recover_value(a_grid, a_star, c_star, l_star, b, g, s, Ptheta)
    % Define Utility function
    U=@(c,l) ((c^g*(1-l)^(1-g))^(1-s))/(1-s);
    %initialize values
    dist=Inf;
    [na, nt]=size(a_star);
    %make initial guess
    V_0=zeros(na,nt);
    for i=1:na 
        for j=1:nt 
            V_0(i,j)=U(c_star(i,j), l_star(i,j));
        end
    end
    %take expectation
    EV=V_0*Ptheta';
    i=0; % start counting iterations
    while dist>1e-4
        Tv=zeros(na, nt);   %Bellman operator
        i=i+1;
        for it=1:nt 
            for ia=1:na
                Tv(ia,it)=U(c_star(ia, it), l_star(ia,it))+...  % initial utility from policy functions
                    b*interp1(a_grid, EV(:,it), a_star(ia,it)); % expected utility from policy functions
                % a_grid is the a_p choice in current period given value
                % a_star. EV is the expected utlity from that transition
            end
        end
        dist=norm(Tv-V_0,Inf);
        V_0=Tv;  %update
        if i>100
            break
        end
    end
    V=Tv;
end
%% Model Solving
function [a,c,l, G, r, w, psi, Y, C] = solve_model(a_grid,logtheta, Pr_theta, tau_c, tau_y, delta, alpha, beta, gamma, sigma)
    %define ub and lb of interest rates
    r_lo=0;
    r_hi=1;
    % initialize distance
    z=Inf;
    i=0;    %keep track of iterations
    while abs(z)>1e-4   % start a bisection search
        i=i+1;   % no longer want this printed
        r_init=(r_lo+r_hi)/2;
        %back out implied capital per labour from initial interest rate
        KLs=((delta+r_init)/alpha)^(1/(alpha-1));
        w_init=(1-alpha)*KLs^(-alpha);
        %already have theta and a grids
        %denote b as benchmark economy
        [ab,cb,lb]=EGM(tau_c, tau_y, beta, sigma, gamma, logtheta, Pr_theta, a_grid, r_init, w_init, 1e-4);
        pb=distribution(a_grid, ab, Pr_theta, 1e-4);
        [Kb,Lb,Cb,Gb, Yb]=aggregates(ab, lb, cb, pb, logtheta, tau_y, tau_c, delta, alpha);

        r_star=alpha*(Kb/Lb)^(alpha-1)-delta;   
        w_star=(1-alpha)*(Kb/Lb)^(-alpha);   % only keeping this because I want the number
        z= Kb/Lb-KLs;   %markets clear if z=0
        fprintf("Solving Model Given tax rates, iteration %d , interest rate is %.4f \n", i, r_star)
    
    %direction of interest rate change
        if z>0  %excess demand for investment
            r_hi=r_init;    %lower the interest rate
        elseif z<0  %excess supply
            r_lo=r_init;    % raise interest rate
        end % last case is markets clear
        if i>40
            break   %want maximum number of iterations
        end

    end
    a=ab;
    c=cb;
    l=lb;
    G=Gb;
    r=r_star;
    w=w_star;
    psi=pb;
    Y=Yb;
    C=Cb;
end

%% Tax Reform
function [a,c,l,r, w, tau_c, psi, Y,C] = solve_reform(a_grid,logtheta, Pr_theta, delta, alpha, beta, gamma, sigma,G)
        %start by initializing grid of potential tau_c solutions
        ntau=10;    %number of potential solutions
        tau_grid=linspace(0,4, ntau);       
        dist=G;
        j=0;    %start the index
        while dist>0 && j<ntau  %This part is becuase I'm assume that government revenue is not strictly increasing in tau_c
            j=j+1; 
            % that is, I expect some sort of laffer curve behavior
            [ar,cr,lr, Gr, rr, wr, psir, Yr, Cr] = solve_model(a_grid,logtheta, Pr_theta, tau_grid(j), 0, delta, alpha, beta, gamma, sigma);
            dist=G-Gr;
            % Want to print this to see what range we wind up in
            fprintf("On grid search iteration %d , Tax rate is %.4f , budget balance is %.4f \n", j,tau_grid(j), dist)
        end % at this point last dist is positive, current dist is negative
        tau_hi=tau_grid(j);
        tau_lo=tau_grid(j-1);
        k=0;
        while abs(dist)>1e-4
            k=k+1;   %want to see how many iterations
            tau_mid=(tau_lo+tau_hi)/2;
            [ar,cr,lr, Gr, rr, wr, psir, Yr, Cr] = solve_model(a_grid,logtheta, Pr_theta, tau_mid, 0, delta, alpha, beta, gamma, sigma);
            dist=G-Gr;
            fprintf("On bisection search iteration %d , Tax rate is %.4f , and budget balance is %.4f \n", k,tau_mid, dist)
            if dist>0
                tau_lo=tau_mid;
            elseif dist<0
                tau_hi=tau_mid;
            end
            if k>40
                break
            end
        end
        a=ar;
        c=cr;
        l=lr;
        r=rr;
        w=wr;
        tau_c=tau_mid;
        psi=psir;
        Y=Yr;
        C=Cr;

end

%% Welfare Analysis
function c=CE1(vb, vr, g, s)
    B=vr-vb;
    exp=(1-s)*g;
    c=(((vr-B)./(vb-B)).^(1/exp)-ones(size(vb)));

    

end

%% Integrate welfare gains

function v=pseudo_int(c, psi)
    %get grid size
    [na, nt]=size(psi);
    % intialize table
    cum_sumc=zeros(6, nt);
    %get breakdown of distribution
    total_psi=sum(psi);
    percentile_upper_bounds=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0];
    for j=1:nt
        k=0;    %index wealth level
        cum_sumt=0; % keep track of CDF
        for i=1:length(percentile_upper_bounds)
            while cum_sumt<percentile_upper_bounds(i) && k<na
                k=k+1;   %increase wealth
                cum_sumt=cum_sumt+psi(k,j)/total_psi(j);  %keep track of CDF of psi
                cum_sumc(7-i,j)=cum_sumc(7-i,j)+c(k,j)*psi(k,j)*100; %add to welfare gain
                fprintf("%d, %d, %0.4f \n", 7-i,j, cum_sumt)
            end
        end
    end
    v=cum_sumc;
end

function v= CE2(vr,vb, psib, psir, g, s)
    e=(1-s)*g;
    B=vr-vb;
    Bbar=sum(sum(B.*psib));
    BBar=Bbar*(ones(size(vb)));
    numerator=sum(sum(psib.*vb-BBar));
    denominator=sum(sum(psir.*vr-BBar));
    v=(numerator/denominator)^(1/e);

end







        









                
                










              



            









