function [pS0] = SCA(gP, gS, p_noise, p_tot, R_targ, Dx, max_count)
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

[~, order] = sort(abs(gS).^2, 1, "descend");
order = order(1: Dx, :);

S = cell(M, 1);
for i = 1: Dx
    for k = 1: K
        S{order(i, k)} = [S{order(i, k)}, k];
    end
end

for count = 1: max_count
cvx_begin
variable pS(M, K) nonnegative
variable z(1, M) nonnegative
expression cross_sec(M, M)
expression interference(M, K)
maximize sum(log(1 + z));

sum(pP) + sum(sum(pow_abs(pS, 2))) <= p_tot;
z0 = z0 + 1e-5;

for k = 1: K
    pS(setdiff(1: M, order(:, k)), k) == 0;
end

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

for i = 1: Dx
    for k = 1: K
        m = order(i, k);
        if i == Dx
            interference(m, k) = 0;
        else
            interference(m, k) = sum(pS(order(i + 1: end, k), k));
        end
    end
end

for m = 1: M
    if isempty(S{m})
        z(m) == 0;
    else
        sum(cross_sec(m, setdiff(1: M, m))) + cross_prim(m) + p_noise <= ...
            cross_sec0(m) / z0(m) + ...
            (pS(m, :) - pS0(m, :)) * (2 * squeeze(G(m, :, :)) * pS0(m, :)' / z0(m)) - ...
            cross_sec0(m) / z0(m)^2 * (z(m) - z0(m));

        for n = 1: length(S{m})
            k = S{m}(n);
            interference(m, k) + pP(k) + p_noise / gP(k)^2 <= ...
                pS0(m, k) / z0(m) - ...
                pS0(m, k) / z0(m)^2 * (z(m) - z0(m)) + ...
                (pS(m, k) - pS0(m, k)) / z0(m);
        end
    end
end
cvx_end

pS0 = full(pS);
z0 = z;
end
end

