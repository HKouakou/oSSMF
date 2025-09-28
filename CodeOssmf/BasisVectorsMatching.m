function match_indices = BasisVectorsMatching(S_true, S_estimate, nb_vectors)

    % Match each estimated pure spectrum with the true one
    % Inputs:
    %   S_true: True pure spectra matrix
    %   S_estimate: Estimated pure spectra matrix
    %   nb_vectors: size(S_true,2)
    
    % Output:
    %   match_indices: indices of the best matches


    % Normalize (l2 norm) each column of S_true and S_estimate
    S_true_normalized = S_true ./ vecnorm(S_true);  
    S_estimate_normalized = S_estimate ./ vecnorm(S_estimate);  

    % Compute the cosine similarity matrix
    cosine_similarity_matrix = S_true_normalized.' * S_estimate_normalized;

    % Find the best match for each column of S_estimate in S_true
    match_indices = zeros(1, nb_vectors);  % Initialize output vector
    for i = 1:nb_vectors
        % Find the maximum cosine similarity for the current column in S_estimate
        [~, max_idx] = max(cosine_similarity_matrix(:, i));
        match_indices(i) = max_idx;

        % Set the corresponding row and column in the similarity matrix to -1
        % to ensure unique matching
        cosine_similarity_matrix(max_idx, :) = -1;
    end
end
