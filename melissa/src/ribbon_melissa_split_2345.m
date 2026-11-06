% MATLAB streamribbons for the validation-selected Melissa reconstruction.
% Seed 2345, best validation epoch 2500, snapshot 20251028T07.
%
% Put this script in melissa/src/ and run it from MATLAB (in melissa/):
%   run('src/ribbon_melissa_split_2345.m')
%
% The input MAT is created by:
%   python src/export_melissa_matlab_split.py
%
% This adapts the earlier ribbon_melissa_reconstructed.m script.
% z exaggeration is VISUAL only: horizontal velocity magnitude uses physical
% U and V. The vertical velocity is NOT an observed atmospheric quantity.

clearvars;
close all;

thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
outDir = fullfile(rootDir, 'outputs', 'split_protocol', 'figures', 'revised_2345');
matFile = fullfile(outDir, 'melissa_reconstructed_for_matlab.mat');

if ~isfile(matFile)
    error(['Cannot find %s\nFirst, from melissa/, run: ' ...
           'python src/export_melissa_matlab_split.py'], matFile);
end
S = load(matFile);

assert(double(S.seed(1)) == 2345, 'Unexpected model seed');
assert(double(S.best_epoch(1)) == 2500, 'Unexpected checkpoint epoch');

xiKm = double(S.xi_m(:).') / 1000;
etaKm = double(S.eta_m(:).') / 1000;
zM = double(S.z_m(:).');

% MATLAB meshgrid(xi,eta,z) arrays are [n_eta,n_xi,n_z].
U = double(S.U_mps);
V = double(S.V_mps);
W = double(S.W_mps);
expected = [numel(etaKm), numel(xiKm), numel(zM)];
assert(isequal(size(U),expected) && isequal(size(V),expected) && ...
       isequal(size(W),expected), 'Unexpected field shape; expected [eta,xi,z]');
assert(all(isfinite(U(:))) && all(isfinite(V(:))) && all(isfinite(W(:))), ...
       'Nonfinite velocity field');

verticalDisplayFactor = 500;
zDisplayKm = zM * verticalDisplayFactor / 1000;
[X,Y,Z] = meshgrid(xiKm, etaKm, zDisplayKm);

% Coordinates are in display km. Streamline direction uses the same
% transform applied to velocity: (U,V,W) --> (U,V,500*W).
% This is equivalent to integrating in physical (xi,eta,z) coordinates
% before applying the vertical display scaling.
Uz = U;
Vz = V;
Wz = W * verticalDisplayFactor;

% The manuscript colors the ribbons by PHYSICAL HORIZONTAL wind speed.
horizontalSpeed = hypot(U, V);
speedMax = max(horizontalSpeed(:));

% Use only starts within the physical grid. The earlier version used
% radius 60/105/150 km and z=60/125/200 m.
seedRadiiKm = [60 105 150];
seedHeightsM = [60 125 200];
[sx,sy,sz] = deal([]);
for level = seedHeightsM
    for radius = seedRadiiKm
        theta = linspace(0,2*pi,13);
        theta(end) = [];
        x0 = radius * cos(theta);
        y0 = radius * sin(theta);
        z0 = level * verticalDisplayFactor / 1000;
        inside = (x0 >= xiKm(1)) & (x0 <= xiKm(end)) & ...
                 (y0 >= etaKm(1)) & (y0 <= etaKm(end)) & ...
                 (z0 >= zDisplayKm(1)) & (z0 <= zDisplayKm(end));
        sx = [sx, x0(inside)]; %#ok<AGROW>
        sy = [sy, y0(inside)]; %#ok<AGROW>
        sz = [sz, ones(1,sum(inside))*z0]; %#ok<AGROW>
    end
end
assert(~isempty(sx), 'No starting positions inside the reconstructed domain');

fig = figure('Color','w','Renderer','opengl', ...
             'Position',[100 100 1200 850]);
ax = axes(fig);
hold(ax,'on');

% Ancho de los listones en las coordenadas de visualizacion (km).
ribbonWidth = 3.0;

h = streamribbon(X,Y,Z,Uz,Vz,Wz,sx,sy,sz,ribbonWidth);

%h = streamribbon(X,Y,Z,Uz,Vz,Wz,sx,sy,sz);
if isempty(h)
    error('MATLAB returned no streamribbons. Try smaller seed radii or other heights.');
end

for k = 1:numel(h)
    set(h(k),'FaceColor','interp','EdgeColor','none','FaceAlpha',0.93);
    hx = get(h(k),'XData');
    hy = get(h(k),'YData');
    hz = get(h(k),'ZData');
    % Color each ribbon using the physical horizontal wind speed.
    c = interp3(X,Y,Z,horizontalSpeed,hx,hy,hz,'linear',NaN);
    set(h(k),'CData',c);
    try
        h(k).AmbientStrength = 0.42;
        h(k).DiffuseStrength = 0.80;
        h(k).SpecularStrength = 0.10;
    catch
    end
end

colormap(ax,jet(256));
caxis(ax,[0,max(speedMax,eps)]);
cb = colorbar(ax);
cb.Label.String = 'Horizontal wind speed [m/s]';

xlabel(ax,'\xi [km]');
ylabel(ax,'\eta [km]');
zlabel(ax,'z [m]');
xlim(ax,[xiKm(1) xiKm(end)]);
ylim(ax,[etaKm(1) etaKm(end)]);
zlim(ax,[zDisplayKm(1) zDisplayKm(end)]);
zLabels = [10 50 100 150 200 250 290];
zLabels = zLabels(zLabels >= min(zM) & zLabels <= max(zM));
zticks(ax,zLabels*verticalDisplayFactor/1000);
zticklabels(ax,cellstr(string(zLabels)));
pbaspect(ax,[1 1 0.6]);
box(ax,'on');
grid(ax,'off');
axis(ax,'vis3d');

view(ax,-55,25);
set(ax,'Projection','perspective','FontSize',11,'LineWidth',0.8);

% Alejar la camara para mostrar mejor el dominio completo.
camzoom(ax,0.75);

%view(ax,-55,25);


set(ax,'Projection','perspective','FontSize',11,'LineWidth',0.8);
shading(ax,'interp');
material(ax,'dull');
camlight(ax,'headlight');
lighting(ax,'gouraud');

title(ax,'Hurricane Melissa | 28 October 2025, 07:00 UTC | seed 2345');

base = fullfile(outDir,'fig_melissa_3d_streamribbons_matlab');
print(fig,base,'-dpdf','-bestfit');
print(fig,base,'-dpng','-r300');
fprintf('MATLAB streamribbons generated:\n  %s.pdf\n  %s.png\n',base,base);
