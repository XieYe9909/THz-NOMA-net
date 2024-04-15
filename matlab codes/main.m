clc;
clear;
close all;

%%
load("data\gP_list.mat");
load("data\gS_list.mat");
[data_num, ~] = size(gP_list);

for i = 1: data_num
    gP = gP_list(i, :);
    gS = squeeze(gS_list(i, :, :));
    [outputArg1, outputArg2] = SCA(gP, gS, 1e-12, 1, 1, 3);
end
