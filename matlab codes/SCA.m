function [outputArg1, outputArg2] = SCA(gP, gS, p_noise, p_tot, R_targ, Dx)
%SCA Summary of this function goes here
%   Detailed explanation goes here
[K, ~] = size(gP);
[M, ~] = size(gS);

pP = (2 ^ R_targ - 1) * p_noise / gP.^2;

pS0 = zeros(M, K);
z0 = zeros(1, M);

[~, order] = sort(abs(gS).^2, 2, "descend");
order = order(:, 1: Dx);

cvx_begin
variable pS(M, K) nonnegative
variable z(1, M) nonnegative
maximize sum(log(1 + z));

sum(pP) + sum(sum(pS.^2)) <= p_tot;

cross_prim = zeros(1, M);
cross_sec = zeros(M, M);
for m = 1: M
    cross_prim(m) = (abs(gS(m, :)).^2) * pP';
    for j = 1: M
        cross_sec(m, j) = abs(gS(m, :) * pS(j, :)').^2;
    end
end


cvx_end

end

