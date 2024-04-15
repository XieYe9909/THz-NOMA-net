function [outputArg1, outputArg2] = SCA(gP, gS, p_noise, p_tot, R_targ, Dx)
%SCA Summary of this function goes here
%   Detailed explanation goes here
norm_ratio = 1e6;
gP = gP * norm_ratio;
gS = gS * norm_ratio;
p_noise = p_noise * norm_ratio^2;

K = length(gP);
[M, ~] = size(gS);

pP = (2 ^ R_targ - 1) * p_noise ./ gP.^2;

pS0 = rand(M, K);
z0 = rand(1, M);

[~, order] = sort(abs(gS).^2, 2, "descend");
order = order(:, 1: Dx);

% cvx_solver mosek
cvx_begin
variable pS(M, K) nonnegative
variable z(1, M) nonnegative
expression cross_sec(M, M)
maximize sum(log(1 + z));

sum(pP) + sum(sum(pow_abs(pS, 2))) <= p_tot;

cross_prim = zeros(1, M);
cross_sec0 = zeros(1, M);
G = zeros(M, K, K);
for m = 1: M
    cross_prim(m) = (abs(gS(m, :)).^2) * pP';
    cross_sec0(m) = abs(gS(m, :) * pS0(m, :)').^2;
    G(m, :, :) = real(gS(m, :)' * gS(m, :));
    for j = 1: M
        cross_sec(m, j) = pow_abs(real(gS(m, :)) * pS(j, :)', 2) + ...
            pow_abs(imag(gS(m, :)) * pS(j, :)', 2);
    end
end


for m = 1: M
    pS(m, setdiff(1: M, order(m, :))) == 0;
    sum(cross_sec(m, setdiff(1: M, m))) + cross_prim(m) + p_noise <= ...
        cross_sec0(m) / z0(m) + ...
        (pS(m, :) - pS0(m, :)) * (2 * squeeze(G(m, :, :)) * pS0(m, :)' / z0(m)) - ...
        cross_sec0(m) / z0(m)^2 * (z(m) - z0(m));
end
cvx_end
end

