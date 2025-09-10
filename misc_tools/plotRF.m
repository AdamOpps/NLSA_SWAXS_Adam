function plotRF(hAxis,x,y,xlabelText,ylabelText,titleText,lineStyle)
% 
% plotRF(hAxis,x,y,xlabelText,ylabelText,titleText,lineStyle)
% 
% produces plots with a consistent look.
% 
% Input:
%   hAxis: use the output of the subplot command.
%   x, y: data to be plotted.
%   xlabelText, ylabelText: x- and y-labels.
%   titleText: plot title.
%   lineStyle: line style, default to be 'b-'.
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This code is provided on the understanding that you and/or your collaborators will
% provide the following acknowledgement in any publication based on results obtained in
% whole or in part using this code in the current or any derivative form:
% 
% "This work was enabled wholly or in part by software provided by Russell Fung and
% collaborators at the University of Wisconsin-Milwaukee."
% 
% Reference:
% "The Symmetries of Image Formation by Scattering. I Theoretical Framework", D. Giannakis,
% P. Schwander, A. Ourmazd, Optics Express, 20, p.12799 (2012).
% 
% "The Symmetries of Image Formation by Scattering. II Applications", P. Schwander,
% D. Giannakis, C.H. Yoon, A. Ourmazd, Optics Express, 20, p.12827 (2012).

  if (nargin==6)
    lineStyle = 'b-';
  end
  ymin = min(y);
  ymax = max(y);
  if (abs(ymax-ymin)<0.1)
    ymin = ymin-0.1;
    ymax = ymax+0.1;
  end
  xmin = min(x);
  xmax = max(x);
  if (abs(xmax-xmin)<0.1)
    xmin = xmin-0.1;
    xmax = xmax+0.1;
  end
  plot(x,y,lineStyle,'lineWidth',2);
  set(hAxis,'xlim',[xmin,xmax],'ylim',[ymin,ymax]);
  set(hAxis,'lineWidth',2,'fontSize',15)
  xlabel(xlabelText,'fontSize',15)
  ylabel(ylabelText,'fontSize',15)
  title(titleText,'fontSize',15)
%end
