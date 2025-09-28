function [M,Up,Up_,D_diag] = SISALmodified(Y,p,Minit,Up_t_minus1,D_t_minus1,y_t,CurrentsampleMean,nb_point,varargin)


%% -------------------------------------------------------------------------
%
% Copyright (May, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%
% SISAL is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] José M. Bioucas-Dias
%     "A variable splitting augmented lagrangian approach to linear spectral unmixing"
%      First IEEE GRSS Workshop on Hyperspectral Image and Signal
%      Processing - WHISPERS, 2009 (submitted). http://arxiv.org/abs/0904.4635v1
%
%
%
% -------------------------------------------------------------------------
%
%% Modifications:
%   Modified by Hugues Kouakou on June, 2025
%     - Removed some parts (verbose, spherization)
%     - Added incremental update of the signal subspace
%     - Replaced inv(.)*x with A\x, and y*inv(.) with y/A 
%%



%% [M,Up,Up_,D_diag] = SISALmodified(Y,p,Minit,Up,D,y_t,CurrentsampleMean,nb_point,varargin)
%
% Simplex identification via split augmented Lagrangian (SISAL)
%
%% --------------- Description ---------------------------------------------
%
%  SISAL Estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional
%  simplex of minimum volume containing the vectors [y_1,...y_N], under the
%  assumption that y_i belongs to a (p-1)  dimensional affine set. Thus,
%  any vector y_i   belongs  to the convex hull of  the columns of M; i.e.,
%
%                   y_i = M*x_i
%
%  where x_i belongs to the probability (p-1)-simplex.
%
%  As described in the papers [1], [2], matrix M is  obtained by implementing
%  the following steps:
%
%   1-Project y onto a p-dimensional subspace containing the data set y
%
%            yp = Up'*y;      Up is an isometric matrix (Up'*Up=Ip)
%
%   2- solve the   optimization problem
%
%       Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:  ones(1,p)*Q=mq,
%
%      where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1])
%   3- Compute
%
%      M = Up*inv(Q^*);
%
%% -------------------- Line of Attack  -----------------------------------
%
% SISAL replaces the usual fractional abundance positivity constraints, 
% forcing the spectral vectors to belong to the convex hull of the 
% endmember signatures,  by soft  constraints. This new criterion brings
% robustnes to noise and outliers
%
% The obtained optimization problem is solved by a sequence of
% augmented Lagrangian optimizations involving quadractic and one-sided soft
% thresholding steps. The resulting algorithm is very fast and able so
% solve problems far beyond the reach of the current state-of-the art
% algorithms.
%
%
%
%%  ===== Required inputs =============
%
% y - matrix with  L(channels) x N(pixels).
%     each pixel is a linear mixture of p endmembers
%     signatures y = M*x + noise,
%
%     SISAL assumes that y belongs to an affine space. It may happen,
%     however, that the data supplied by the user is not in an affine
%     set. For this reason, the first step this code implements
%     is the estimation of the affine set the best represent
%     (in the l2 sense) the data.
%
%  p - number of independent columns of M. Therefore, M spans a
%  (p-1)-dimensional affine set.
%   
%  Minit - initialization, size Lxp 
%
%  Up_t_minus1 = Up_ (computed at iteration t-1)
%
%  D_t_minus1 = D_diag (computed at iteration t-1)
%
%  y_t - new observation at iteration t
%
%  CurrentsampleMean - sample mean at iteration t (mean of y_1,...,y_t)
%
%  nb_point - (= t, number of observations at iteration t)

%%  ====================== Optional inputs =============================
%
%  'MM_ITERS' = double; Default 80;
%
%               Maximum number of constrained quadratic programs
%
%
%  'TAU' = double;
%
%               Regularization parameter in the problem
%
%               Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:ones(1,p)*Q=mq,
%
%              where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1]).
%
%  'MU' = double; 
%
%              Augmented Lagrange regularization parameter
%
%
%
%
%%  =========================== Outputs ==================================
%
% M  =  [Lxp] estimated mixing matrix
%
% Up =  [Lxp] isometric matrix spanning  the same subspace as M
%
% Up_ = Up[:,1:p-1]
%
% D_diag  = (p-1) eigenvalues of the sample covariance matrix. The dynamic range
%                 of these eigenvalues gives an idea of the  difficulty of the
%                 underlying problem
%
%
% NOTE: the identified affine set is given by
%
%              {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
%
%% -------------------------------------------------------------------------
%
% Copyright (May, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%
% SISAL is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] José M. Bioucas-Dias
%     "A variable splitting augmented lagrangian approach to linear spectral unmixing"
%      First IEEE GRSS Workshop on Hyperspectral Image and Signal
%      Processing - WHISPERS, 2009 (submitted). http://arxiv.org/abs/0904.4635v1
%
%
%
% -------------------------------------------------------------------------
%
%% Modifications:
%   Modified by Hugues K. on June, 2025
%     - Removed some parts (verbose, spherization)
%     - Added incremental update of the signal subspace
%     - Replaced inv(.)*x with A\x, and y*inv(.) with y/A in some parts
%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 8
    error('Wrong number of required parameters');
end
% data set size
[L,N] = size(Y);
if (L<p)
    error('Insufficient number of columns in Y');
end
%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% maximum number of quadratic QPs
MMiters = 80; 
% soft constraint regularization parameter
tau = 0.3;  
mu = 1.4; 



% initial simplex
M = Minit;



%%

% quadractic regularization parameter for the Hesssian
lam_quad = 1e-6; 
% minimum number of AL iterations per quadratic problem 
AL_iters = 4;  



%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MM_ITERS'
                MMiters = varargin{i+1};
            case 'MU'
                mu = varargin{i+1};
            case  'TAU'
                tau = varargin{i+1};
            case 'M0'
                M = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end


%%
%--------------------------------------------------------------
% identify the affine space that best represent the data set Y 
%--------------------------------------------------------------

if nb_point==0
     my = CurrentsampleMean;
    Y = Y-repmat(my,1,N);
    [Up,D] = svds(Y*Y'/N,p-1);
else
    [Up,D] = UpdateSubspace(Up_t_minus1, CurrentsampleMean, D_t_minus1, y_t,nb_point,p);
    my=CurrentsampleMean;
    Y = Y-repmat(my,1,N);
end

% represent Y in the subspace R^(p-1)
Y = Up*Up'*Y;


Up_ = Up; 
D_diag = D;


% lift Y
Y = Y + repmat(my,1,N);   
% compute the orthogonal component of my
my_ortho = my-Up*Up'*my;
% define another orthonormal direction
Up = [Up my_ortho/sqrt(sum(my_ortho.^2))];


% get coordinates in R^p
Y = Up'*Y;


%%
% ---------------------------------------------
%            Initialization
%---------------------------------------------
if M == 0
    % Initialize with VCA
    Mvca = VCA(Y,'Endmembers',p,'verbose','off');
    M = Mvca;
    % expand Q
    Ym = mean(M,2);
    Ym = repmat(Ym,1,p);
    dQ = M - Ym; 
    % fraction: multiply by p is to make sure Q0 starts with a feasible
    % initial value.
    M = M + p*dQ;
else
 
    M = M-repmat(my,1,p);
    M = Up(:,1:p-1)*Up(:,1:p-1)'*M;
    M = M +  repmat(my,1,p);
    M = Up'*M;   % represent in the data subspace
 
end
Q0 = inv(M);
Q=Q0;


%%
% ---------------------------------------------
%            Build constant matrices
%---------------------------------------------

AAT = kron(Y*Y',eye(p));    % size p^2xp^2
B = kron(eye(p),ones(1,p)); % size pxp^2
qm = sum((Y*Y')\Y,2);


H = lam_quad*eye(p^2);
F = H+mu*AAT;          % equation (11) of [1]
IF = inv(F);

% auxiliar constant matrices
G = (F\B')/(B*(F\B'));
qm_aux = G*qm;
G = IF-(G*B)/F;

%%
% ---------------------------------------------------------------
%          Main body- sequence of quadratic-hinge subproblems
%----------------------------------------------------------------

% initializations
Z = Q*Y;
Bk = 0*Z;


for k = 1:MMiters
    
    IQ = inv(Q);
    g = -IQ';
    g = g(:);

    baux = H*Q(:)-g;

    q0 = Q(:);
    Q0 = Q;
    
  
    if k==MMiters
        AL_iters = 100;
    end
    
    
    while 1 > 0
        q = Q(:);
        % initial function values (true and quadratic)
        f0_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        f0_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        for i=2:AL_iters
            %-------------------------------------------
            % solve quadratic problem with constraints
            %-------------------------------------------
            dq_aux= Z+Bk;             % matrix form
            dtz_b = dq_aux*Y';
            dtz_b = dtz_b(:);
            b = baux+mu*dtz_b;        % (11) of [1]
            q = G*b+qm_aux;           % (10) of [1]
            Q = reshape(q,p,p);
            
            %-------------------------------------------
            % solve hinge
            %-------------------------------------------
            Z = soft_neg(Q*Y -Bk,tau/mu);
                     
            %-------------------------------------------
            % update Bk
            %-------------------------------------------
            Bk = Bk - (Q*Y-Z);
        end
        f_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        if f0_quad >= f_quad    % quadratic energy decreased
            while  f0_val < f_val
                % do line search
                Q = (Q+Q0)/2;
                f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
            end
            break
        end
    end

end

M = Up/Q;
end

function z = hinge(y)
%   hinge function
z = max(-y,0);
end

function z = soft_neg(y,tau)
%  negative soft (proximal operator of the hinge function)
z = max(abs(y+tau/2) - tau/2, 0);
z = z./(z+tau/2) .* (y+tau/2);
end



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %