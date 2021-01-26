\PassOptionsToPackage{fit}{truncate}
\ProvidesClass{numapde-thesis}[2020/03/17]
% Part of this work has been inspired by the jnsao.cls file
% from http://jnsao.episciences.org

% TODO
% Hack to include statement of authorship
\pdfminorversion=6

% Declare the options for the class
% Turn on line numbering
\DeclareOption{lineno}{\AtBeginDocument{\linenumbers}}

% Pass all non-implemented options to the base class scrartcl
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrreprt}}

% Process all options
\ProcessOptions\relax

% Load the base class scrreprt with default documentclass option english
% https://tex.stackexchange.com/questions/147243/a-class-with-default-language-settings
\RequirePackage{etoolbox}
\preto\@classoptionslist{english,}
\LoadClass[a4paper,DIV=12,parskip=full,headings=optiontohead]{scrreprt}

% Resolve the dependencies of this package
\RequirePackage{numapde-cls-commons}
\RequirePackage{lineno}
\RequirePackage{pdfpages}

% Configure the document class specific settings
% Set document type
\def\numapde@documenttype{Thesis}

\let\numapde@thesis\empty
\newcommand{\thesis}[1]{\gdef\numapde@thesis{#1}}
\let\numapde@degree\empty
\newcommand{\degree}[1]{\gdef\numapde@degree{#1}}
\let\numapde@studntID\empty
\newcommand{\studentID}[1]{\gdef\numapde@studentID{#1}}
\let\numapde@submissiondate\empty
\newcommand{\submissiondate}[1]{\gdef\numapde@submissiondate{#1}}
\let\numapde@firstsupervisor\empty
\newcommand{\firstsupervisor}[1]{\gdef\numapde@firstsupervisor{#1}}
\let\numapde@secondsupervisor\empty
\newcommand{\secondsupervisor}[1]{\gdef\numapde@secondsupervisor{#1}}
\let\numapde@supervisornote\empty
\newcommand{\supervisornote}[1]{\gdef\numapde@supervisornote{#1}}
\let\numapde@logo\empty
\newcommand{\logo}[1]{\gdef\numapde@logo{#1}}
\logo{logo/TU_Chemnitz_Logo_gruen_RGB.png}
% Set up the appearance of author, date, title, and section headings etc.
\renewcommand*{\sectfont}{\color{TolMutedBlue}\sffamily}
\setkomafont{author}{\large\normalfont\sffamily}
\setkomafont{date}{\large\normalfont\sffamily}
\setkomafont{title}{\color{TolMutedBlue}\normalfont\sffamily\scshape}
\addtokomafont{disposition}{\scshape}
\addtokomafont{subsubsection}{\scshape\itshape}

% Remove \@date and spacing following it from \@maketitle
% https://tex.stackexchange.com/questions/288800/omit-the-date-in-maketitle-without-blanking-the-date
\patchcmd{\@maketitle}{{\usekomafont{date}{\@date \par}}\vskip \z@ \@plus 1em}{}{}{}

% Set up the page header and footer
\addtokomafont{pageheadfoot}{\normalfont\sffamily\footnotesize}
\if@twoside
	\refoot[]{\truncate{1.0\textwidth}{\@author}}
	\lofoot[]{\truncate{1.0\textwidth}{\ifx\numapde@shorttitle\empty{\@title}\else{\numapde@shorttitle}\fi}}
	\ofoot[{\GetTranslation{page}~\thepage~\GetTranslation{of}~\pageref*{LastPage}}]{\GetTranslation{page}~\thepage~\GetTranslation{of}~\pageref*{LastPage}}
\else
	\chead[]{\truncate{0.45\textwidth}{\ifx\numapde@shorttitle\empty{\@title}\else{\numapde@shorttitle}\fi}}
	\ifoot[]{\truncate{0.45\textwidth}{\@author}}
	\ofoot*{\GetTranslation{page}~\thepage~\GetTranslation{of}~\pageref*{LastPage}}
\fi
\cfoot*{}

\renewcommand{\maketitle}{%
	\begin{titlepage}
		\begin{minipage}{59mm}\centering
			\includegraphics[width=47mm]{\numapde@logo}
		\end{minipage}
		\begin{minipage}{2mm}\centering
			\tikz\draw[black] (0,0) -- (0,-2.5);
		\end{minipage}
		\hspace{1cm}
		\begin{minipage}{\textwidth-72mm}\large
			Technische Universität Chemnitz \\
			Fakultät für Mathematik \\
			Professur Numerische Mathematik \\
			(Partielle Differentialgleichungen)
		\end{minipage}
		\vspace{3cm}
		\begin{center}
			\vspace{1cm}
			{\huge\@title{}}\\
			\vspace{1cm}
			Computerpraktikum
		\end{center}
		\vfill
		\begin{tabular}[b]{p{0.33\textwidth}l}
			\GetTranslation{submitted by} & \@author\\%
			\GetTranslation{student number} & \numapde@studentID\\%
			\GetTranslation{referee} & \numapde@firstsupervisor\\%
			& \numapde@supervisornote\\
			\GetTranslation{date of submission} & \numapde@submissiondate
		\end{tabular}
	\end{titlepage}
}
