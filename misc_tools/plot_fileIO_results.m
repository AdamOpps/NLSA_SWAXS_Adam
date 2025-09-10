cxfel_root = getenv('CXFEL_ROOT')
addpath([cxfel_root '/misc_tools'])

num_float = []; % in billions
file_size = []; % in GB
t_write   = []; % in minutes
t_read    = []; % in minutes
num_float = [num_float,10.00,20.00,30.00,40.00,42.50,43.75,44.38,44.69,44.73]; %
file_size = [file_size,   75,  150,  224,  299,  317,  326,  331,  333,  334]; % 01/21-22/2024
t_write   = [t_write  ,    2,    4,    7,    9,    9,    9,    9,    9,   12]; %
t_read    = [t_read   ,    8,   15,  NaN,   24,   62,   70,  NaN,   62,   83]; % NaN replaced 3 & 9
num_float = [num_float,40.00,42.50,43.75,44.38,44.69,44.73]; %
file_size = [file_size,  299,  317,  326,  331,  333,  334]; % 01/24/2024
t_write   = [t_write  ,    8,    8,    9,   10,   10,   10]; %
t_read    = [t_read   ,   47,   59,   59,   65,   78,   83]; %
num_float = [num_float,10.0,20.0,30.0,40.0]; %
file_size = [file_size,  75, 150, 224, 299]; % 01/25/2024
t_write   = [t_write  ,   3,   4,   6,   9]; %
t_read    = [t_read   ,   8,  22,  34, NaN]; % NaN replaced 13
num_float = []; % in billions
file_size = []; % in GB
t_write   = []; % in minutes
t_read    = []; % in minutes
num_float = [num_float,10.0,20.0,30.0,40.0,50.0,60.0];     %
file_size = [file_size,  75, 150, 224, 299, 373, 448];     % 02/01/2024
t_write   = [t_write  ,[ 93, 178, 277, NaN, 461, 612]/60]; % NaN replaced 781
t_read    = [t_read   ,[365,1263,2587,4030,5588,9434]/60]; %
num_float = [num_float, 65.00000,66.25000,66.87500,67.18750];     %
file_size = [file_size,      485,     494,     499,     501];     % 02/03/2024
t_write   = [t_write  ,[     656,     625      614,     619]/60]; %
t_read    = [t_read   ,[    7811,    9102,    9936,    7610]/60]; %

hFigure = figure(1);
set(hFigure,'color','w')
set(hFigure,'resize','off')
Pix_SS = get(0,'screensize');
screenWidth = Pix_SS(3);
screenHeight = Pix_SS(4);
pos = [10 900 screenWidth/4 screenHeight/4];
try
  warning('off','Octave:abbreviated-property-match')
catch
end
set(hFigure,'pos',pos)

figure(hFigure)

hsp = subplot(1,1,1);
my_title = 'file I/O runtime on execute-4003.mortimer';
my_xlabel = '#float (billions)';
my_ylabel = 'runtime (min.)';
plotRF(hsp,num_float,t_write,my_xlabel,my_ylabel,my_title,'b<')
addplotRF(hsp,num_float,t_read,'r>')
legend({'writing','reading'},'location','north','orientation','horizontal','box','off');

myJPEG = sprintf('IO_runtime_on_execute_4003.jpg');
print(myJPEG)
