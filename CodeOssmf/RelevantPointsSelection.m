function  [new_relevant_points,update_required_vertices]= RelevantPointsSelection(Up, currentVertices,CurrentsampleMean,Up_,y_t,eps1,eps2,eta,d,relevant_points)
    
    % Inputs
    %  Up (Lxp), Up_ (Lxp-1): basis matrix (signal subspace)
    %  currentVertices: current vertices
    %  CurrentsampleMean: sample mean at iteration t (mean of y_1,...,y_t)
    %  y_t: new observation at iteration t
    %  eps1, eps2, eta, d: proximity parameters (see paper) 
    %  relevant_points: relevant points (at iteration t-1)

    % Outputs
    %  new_relevant_points: relevant points (iteration t)
    %  update_required_vertices: 0 = not required , 1 = required
  


    V = [relevant_points y_t];
    V1 = relevant_points;
    V = Up_*Up_'*(V-CurrentsampleMean); % Noise reduction
    Vr = Up'*(V+CurrentsampleMean); % Reduced version of V

    Sr = Up'*currentVertices ; % Reduced version of the estimated vertices (iteration t-1), size pxp (p = rank of factorization)

    % Estimation of the coefficients w.r.t. the estimated vertices (iteration t-1)
    C = Sr\Vr; 
            
    % Preliminary test
    c_t = C(:,end);
    [update_required_vertices,is_within_facet_tolerance]= PreliminaryTest(c_t,eps1,eps2,eta);
        

    % 
    C=C(:,1:end-1);
    % Condition 1: nonnegativity (tolerance eps1)
    condition1 = all(C >= -eps1, 1);
      
    % Condition 2: sum-to-one (tolerance eps2) 
    sum_rows = sum(C, 1);
    condition2 = (1 - eps2 <= sum_rows) & (sum_rows <= 1 + eps2);
    
    
    % Find indices
    indices = find((condition1 & condition2)==0);
    firstSelection = V1(:,indices);

    % Find indices 
    indices = find((condition1 & condition2)==1);
    secondSelection = V1(:,indices);
    C = C(:,indices);

    %condition3: near a facet (tolerance eta)
    condition3 =   min(C,[],1) <= eta; 
    indices = find(condition3==1);

    
    new_relevant_points = [firstSelection secondSelection(:,indices)]; 

     % Test whether y_t should be added to new_relevant_points (redundancy reduction)
     if update_required_vertices==0
         if is_within_facet_tolerance==1
             C=C(:,indices);
             ind_same_facet=find(sum(abs((C<=eta) - (c_t<=eta)),1)==0.) ;
             ind_diff = setdiff(1:size(C,2),ind_same_facet);
             if ~isempty(ind_same_facet)
                 min_d = min(vecnorm(C(:,ind_same_facet) - c_t));
                 if min_d >=d
                      new_relevant_points = [new_relevant_points y_t];
                 end
             else
                  min_d = min(vecnorm(C(:,ind_diff) - c_t));
                 if min_d >=d
                      new_relevant_points = [new_relevant_points y_t];
                 end
             end

         end %end if is_within_facet_tolerance==1
     else
          new_relevant_points = [new_relevant_points y_t];
     end 
end



function  [update_required_vertices,is_within_facet_tolerance] = PreliminaryTest(C,eps1, eps2,eta)
 
    % Inputs
    %  C: coefficient vector
    %  eps1, eps2, eta: proximity parameters (see paper) 

    % Outputs
    %  update_required_vertices: 0 = not required , 1 = required
    %  is_within_facet_tolerance: 0 = no , 1 = yes


    is_within_facet_tolerance = 0;
    update_required_vertices = 1;
    
    sum_rows = sum(C, 1);
    
    if all(C >= -eps1) && (1 - eps2 <= sum_rows) && (sum_rows <= 1 + eps2)
        update_required_vertices = 0;
        if min(C,[],1) <= eta
            is_within_facet_tolerance = 1;
        end
    end
    
end
