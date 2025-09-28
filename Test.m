
close('all'); 


root_path = fileparts(mfilename('fullpath'));
addpath(fullfile(root_path, 'CodeOssmf'));

data_path = fullfile(root_path, 'Data');

%-----------------------------------------------------------------------

    
% H = coefficient matrix: select file 'coeff_3': k=3 or 'coeff_7': k=7
% k = number of vertices (factorization rank)

% maximum purity level = 0.7 for coeff_3 and coeff_7
H = load(fullfile(data_path, 'coeff_7.csv')); 


% Basis vectors, the first 7 were used in the paper
w = (load(fullfile(data_path, 'basisVectors.mat')).signals)'; 

 
%
 H_size = size(H);
 k = H_size(2); % rank(basis_matrix)
 N = H_size(1); % number of observations
 L = size(w,2); % dimension (original space)

 % Randomization of the rows of H each time the script is executed
 H = H(randperm(N),:);


% Generate data matrix 

SNR = 15; % in dB, change to test different noise levels

indices_spectres = 1:k;
w = w(indices_spectres,:);
Y0 = H*w ; % Noiseless observations, NxL
variance = sum(Y0(:).^2)/10^(SNR/10)/N/L; 
Noise = sqrt(variance).*randn([L N]).';
Y = max(0,Y0 + Noise).'; % Y>=0

%%

n_init = 30;

relevant_points = Y(:,1:n_init) ; % Initialization
CurrentsampleMean = mean(relevant_points,2);


% initialization: first estimates of the vertices
[currentVertices,Up,Up_,D_diag] = SISALmodified(relevant_points,k,0,0,0,0,CurrentsampleMean,0);


% proximity parameters (Refer to the paper for a discussion of these parameters)
eps1 = 1e-4; % small positive value
eps2 = 1e-4; % small positive value
eta = 0.03;  % [0,1]
d = 0.7; % [0,1]


N_str = num2str(N);

for t=n_init+1:N

    disp(['Iteration ',num2str(t),'/',N_str])

    y_t = Y(:,t);
    [relevant_points,update_required_vertices]= RelevantPointsSelection(Up, currentVertices,CurrentsampleMean,Up_,y_t,eps1,eps2,eta,d,relevant_points);

    % Update the sample mean
    CurrentsampleMean = (1/t)*(y_t - CurrentsampleMean) + CurrentsampleMean;

    if update_required_vertices == 1
         [currentVertices,Up,Up_,D_diag] = SISALmodified(relevant_points,k,currentVertices,Up_,D_diag,y_t,CurrentsampleMean,t);       
    end
 end
    


%%
w=w.'; % True vertices
% Set negative values to zero
estimatedVertices = max(currentVertices,0); % Estimated vertices (last iteration)

% 
disp(['total number of observations = ',N_str])
disp(['number of relevant observations at iteration ',N_str,' = ', num2str(size(relevant_points,2))])


%%
% 2D visualization: full observations, relevant observations at final iterate (N),
% and true vs. estimated vertices at the final iteration

center = sum(Y, 2)/N; 
Y_center = Y - center;
[U,S,~]=svd((Y_center*Y_center')/(N-1)); 
A = U(:,1:2);
relevant_points_reduced = (A.'*(relevant_points -center)).';


figure;
axe1 = relevant_points_reduced(:,1);
axe2 = relevant_points_reduced(:,2); 
plot(axe1, axe2, 'o','DisplayName','relevantPoints(LastIteration)',LineWidth=2, Color=[0.5 0 0.5]);


% True vertices
hold on
w_reduced = (A'*(w -center))';
plot(w_reduced(:,1), w_reduced(:,2), 'o','DisplayName','TrueVertices',LineWidth=2, Color='blue');


% Estimated vertices
hold on
estimatedVertices_reduced = (A'*(estimatedVertices -center))';
plot(estimatedVertices_reduced(:,1), estimatedVertices_reduced(:,2), 'o','DisplayName','EstimatedVertices(LastIteration)',LineWidth=2,Color='red');

hold off
legend('Location','best')
set(gca,'XTick',[], 'YTick', [])
title("2D visualization")


figure;
Y_reduced = (A.'*Y_center).';
axe1 = Y_reduced(:,1);
axe2 = Y_reduced(:,2); 
plot(axe1, axe2, 'o','DisplayName','AllObservations',LineWidth=2, Color=[0.5 0 0.5]);


% True vertices
hold on
w_reduced = (A'*(w -center))';
plot(w_reduced(:,1), w_reduced(:,2), 'o','DisplayName','TrueVertices',LineWidth=2, Color='blue');


% Estimated vertices
hold on
estimatedVertices_reduced = (A'*(estimatedVertices -center))';
plot(estimatedVertices_reduced(:,1), estimatedVertices_reduced(:,2), 'o','DisplayName','EstimatedVertices(LastIteration)',LineWidth=2,Color='red');


hold off
legend('Location','best')
set(gca,'XTick',[], 'YTick', [])
title("2D visualization")

% Profiles of basis vectors (estimated / true)
match_indices = BasisVectorsMatching(w, estimatedVertices,k);

figure;
lwidth=2;


for u=1:k
    subplot(2, k, u);
    plot(w(:,match_indices(u)),Color='blue',LineWidth=lwidth)
    if(u==1)
        ylabel('True','FontSize',12)
    end
    set(gca,'XTick',[])
    subplot(2, k, u+k);
    plot(estimatedVertices(:,u),Color='red',LineWidth=lwidth)
    if(u==1)
        ylabel('Estimated(LastIteration)','FontSize',12)
    end
end
















