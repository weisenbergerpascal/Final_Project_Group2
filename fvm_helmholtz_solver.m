%% ------------- Parameters ----------------
epsdi = 1; epsbg = 1;   % uniform permittivity
omega = 15;             % angular frequency
Jz = 1;                  % source amplitude

Lx = 2; Ly = 2;      % domain size
lambda = 2*pi/omega;
dx = lambda/60; dy = lambda/60;          % grid spacing

xvals = 0:dx:Lx;
yvals = 0:dy:Ly;
Nx = numel(xvals); Ny = numel(yvals);

[X,Y] = meshgrid(xvals,yvals);
Rgrid = (X-Lx/2).^2 + (Y-Ly/2).^2;

epsr = epsbg*ones(Ny,Nx);
epsr(Rgrid < 800) = epsdi;


% Source location
ixs = round(Nx/2);%round(Nx/2);
iys = round(Ny/2);%round(Ny/2);

% Gaussian source width (in cells)
sigma = 0.01/dx;

% ------------- Build Gaussian source ----------------
[Xc,Yc] = meshgrid(1:Nx,1:Ny);
b = exp(-((Xc-ixs).^2 + (Yc-iys).^2)/(2*sigma^2));
b = -1i*omega*Jz*b/max(b(:));
%b = zeros(Ny,Nx);
%b(iys,ixs) = -1i*omega*Jz;

%b(imag(b)>-300) = 0;
S = imag(b);

b = b(:); % flatten to vector

% ------------- Loss to absorb waves exiting the domain ----------------
pml_thickness = 0.25*min(Lx,Ly);
pml_strength = 1;

loss = ones(Ny, Nx);
for i = 1:Nx
    for j = 1:Ny
        % distance from boundary in cells
        dist = min([dx*(i-1), dx*(Nx-i), dy*(j-1), dy*(Ny-j)]);
        if dist < pml_thickness
            loss(j,i) = pml_strength*exp(-((pml_thickness-dist)/pml_thickness)^2 * 3);
        end
    end
end

epsr_eff = epsr .* (1 - 1i*0.5*(1 - loss));

% ------------- 9-point Laplacian weights ----------------
cx = 1/dx^2; cy = 1/dy^2;
alpha = 4/6;  % axial neighbors
beta  = 1/6;  % diagonal neighbors

N = Nx*Ny;
main  = zeros(N,1);
east  = zeros(N,1);
west  = zeros(N,1);
north = zeros(N,1);
south = zeros(N,1);
ne = zeros(N,1);
nw = zeros(N,1);
se = zeros(N,1);
sw = zeros(N,1);

for j=1:Ny
    for i=1:Nx
        idx = (j-1)*Nx + i;
        er = epsr_eff(j,i);
        main(idx) = -2*alpha*(cx+cy) - 4*beta*(cx+cy) + er*omega^2;

        % Axial neighbors
        if i>1, west(idx)=alpha*cx; end
        if i<Nx, east(idx)=alpha*cx; end
        if j>1, south(idx)=alpha*cy; end
        if j<Ny, north(idx)=alpha*cy; end

        % Diagonal neighbors
        if i>1 && j>1, nw(idx)=beta*(cx+cy); end
        if i<Nx && j>1, ne(idx)=beta*(cx+cy); end
        if i>1 && j<Ny, sw(idx)=beta*(cx+cy); end
        if i<Nx && j<Ny, se(idx)=beta*(cx+cy); end
    end
end

% build Laplacian
Lap = spdiags(main,0,N,N);

% Masks to remove neighbors at boundaries
east_mask  = true(N,1); east_mask(Nx:Nx:end)=false;
west_mask  = true(N,1); west_mask(1:Nx:end)=false;
north_mask = true(N,1); north_mask((end-Nx+1):end)=false;
south_mask = true(N,1); south_mask(1:Nx)=false;

ne_mask = east_mask & north_mask;
nw_mask = west_mask & north_mask;
se_mask = east_mask & south_mask;
sw_mask = west_mask & south_mask;

% Axial neighbors
Lap = Lap + spdiags(east.*east_mask,1,N,N);
Lap = Lap + spdiags(west.*west_mask,-1,N,N);
Lap = Lap + spdiags(north.*north_mask,Nx,N,N);
Lap = Lap + spdiags(south.*south_mask,-Nx,N,N);

% Diagonal neighbors
Lap = Lap + spdiags(ne.*ne_mask,Nx+1,N,N);
Lap = Lap + spdiags(nw.*nw_mask,Nx-1,N,N);
Lap = Lap + spdiags(se.*se_mask,-Nx+1,N,N);
Lap = Lap + spdiags(sw.*sw_mask,-Nx-1,N,N);

% boundary
% bottom nodes (j=1): indices 1:Nx
bottom_idx = 1:Nx;
% top nodes (j=Ny): indices (Ny-1)*Nx + (1:Nx)
top_idx = (Ny-1)*Nx + (1:Nx);
% left nodes (i=1): indices 1:Nx:N
left_idx = 1:Nx:N;
% right nodes (i=Nx): indices Nx:Nx:N
right_idx = Nx:Nx:N;

% diagonal additions:  i*omega / dn  (dn = dx or dy)
vals_bottom = 1i*omega / dy * ones(numel(bottom_idx),1);
vals_top    = 1i*omega / dy * ones(numel(top_idx),1);
vals_left   = 1i*omega / dx * ones(numel(left_idx),1);
vals_right  = 1i*omega / dx * ones(numel(right_idx),1);

% combine (avoid double-counting corners by summing later)
rows = [bottom_idx, top_idx, left_idx, right_idx];
cols = rows;
vals = [vals_bottom; vals_top; vals_left; vals_right];

% corners appear twice — that's fine because the ghost-point derivation
% contributes from both coordinate directions; if you prefer you can combine
% duplicates by using accumarray, e.g.:
[uniqueIdx, ~, ic] = unique(rows);
vals_combined = accumarray(ic(:), vals(:));
BC = sparse(uniqueIdx, uniqueIdx, vals_combined, N, N);

% add BC correction to Lap
Lap = Lap + BC;
E = Lap\b;
Efield = reshape(E,Ny,Nx);

% ------------- Plot ----------------
figure;
subplot(1,3,1);
imagesc(xvals,yvals,real(Efield)); axis equal tight;
title('Re(E_z)'); colorbar;
subplot(1,3,2);
imagesc(xvals,yvals,imag(Efield)); axis equal tight;
title('Im(E_z)'); colorbar;
subplot(1,3,3);
imagesc(xvals,yvals,abs(Efield)); axis equal tight;
title('Abs(E_z)'); colorbar;
set(gcf,'color','w')


%% plot of permittivity and Jz

figure
subplot(1,2,1);
imagesc(xvals,yvals,imag(epsr_eff)); axis equal tight;
title('\epsilon (permittivity)'); colorbar;
subplot(1,2,2);
imagesc(xvals,yvals,reshape(imag(b),[Ny,Nx])); axis equal tight;
title('J_z (source)'); colorbar;
set(gcf,'color','w')

%% save svg plots
print('-painters','-dsvg',"/home/kholt/workspace/kronosai/waveguide")



%% output data for use with pinn
data = zeros(Nx*Ny,7);
data(:,1) = X(:);
data(:,2) = Y(:);
data(:,3) = real(epsr_eff(:));
data(:,4) = imag(epsr_eff(:));
data(:,5) = imag(b);
data(:,6) = real(Efield(:));
data(:,7) = imag(Efield(:));

writematrix(data,'~/workspace/kronosai/simdata_centered.csv')


%% write image
% imwrite(real(Efield),'~/Downloads/real.png')
% imwrite(imag(Efield),'~/Downloads/imag.png')
% imwrite(imag(b),'~/Downloads/src.png')
% imwrite(real(epsr_eff),'~/Downloads/perm_real.png')
% imwrite(-imag(epsr_eff),'~/Downloads/perm_imag.png')
Ereal = real(Efield);
Eimag = imag(Efield);
eps_real = real(epsr_eff);
eps_imag = imag(epsr_eff);
save('~/Downloads/siren_data.mat','S','Ereal','Eimag','eps_real','eps_imag')
