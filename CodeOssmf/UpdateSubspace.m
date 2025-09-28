% refs. 
% R. Arora, A. Cotter, K. Livescu, and N. Srebro, “Stochastic optimization for pca and pls,” in Proc. Annual Allerton Conf. Comm., Control
    %and Computing (Allerton), 2012, pp. 861–868.
% H. Cardot, D. Degras, "Online principal component analysis in high dimension: Which algorithm to choose?.”  International Statistical Review, 86(1), 29-50.

function [U_update,D_update] = UpdateSubspace(U_t, CurrentsampleMean, D_t, y_t,nb_point,p)

        %Inputs:
        % U_t: current basis matrix (signal subspace)
        % CurrentsampleMean: sample mean at iteration t (mean of y_1,...,y_t)
        % D_t: singular values from the previous svd (line 27)
        % y_t: new observation at iteration t
        % nb_point: (= t, number of observations at iteration t)
        % p: factorization rank

        %Outputs:
        % U_update: updated basis matrix (signal subspace)
        % D_update: updated singular values
        

        center_t = y_t - CurrentsampleMean; % Lx1
        coord = U_t.'*center_t; % (p-1)x1
        center_t_proj = center_t - U_t*pinv(U_t)*center_t; %Lx1
        norm_center_t_proj = norm(center_t_proj);
        myU = norm_center_t_proj*coord;
        Q = ((nb_point-1)/(nb_point^2))*[nb_point*D_t+coord*coord.' myU; myU.'  norm_center_t_proj^2 ]; % pxp
        [U, D, ~] = svd(Q); % U: pxp, D: pxp
        U = [U_t center_t_proj /norm_center_t_proj]*U; % Lxp

        % update
        % basis matrix
        U_update = U(:,1:p-1);% Lx(p-1)
        % singular values
        D_update = D(1:p-1,1:p-1);
end