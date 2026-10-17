"""Standalone Liar's Dice bot (2-10 players).

One file, Python standard library only. Trained 30,000-game neural network is
embedded below as compressed float16 weights. Rules follow the provided engine:
ones are wild, revealed dice last only this round, ones -> normals uses 2*n.

Decision method:
  1) Compute exact (under the dice prior) immediate CALL value using neural V.
  2) Shortlist BIDs by a joint dice/challenge probability model.
  3) Screen distinct REVEAL subsets by exact distributions of rerolled counts.
  4) Adaptive common-seed continuations of the current round for promising moves.
  5) Score end-of-round states by predicted expected final placement.

The training population and continuation policies approximate unknown rivals;
this is not a game-theoretically exact solution. The standalone bot exposes
only decide(s) to the tournament engine and needs no external model files.
"""
import base64
from functools import lru_cache
from itertools import product
import math
from operator import mul
import random
import struct
import time
import zlib

# Safety margin for the original engine's three-second turn limit.
TIME_LIMIT = 2.30
# Candidate counts are capped for expensive 10-player, high-dice positions.
MAX_BID_CANDIDATES = 16
MAX_REVEAL_CANDIDATES = 5
MAX_REVEAL_SCREEN = 65
# Numbers of shared-seed rollout samples per candidate at each screening stage.
ROLLOUT_STAGES = ((18, None), (55, 6), (150, 3))
# Slight evidence tilt toward faces a bidder previously claimed. Set 0 to disable.
HISTORY_WEIGHT = 0.16
# Softness in preliminary challenger-behavior model (NOT the rollout policy).
FAST_CALL_SHARPNESS = 1.9

# Compressed, pretrained PyTorch weights (23-96-96-10, float16).
# Embedded so a single .py file is sufficient, even without numpy/torch.
_WEIGHTS_B85 = (
    'c-'
    'j<~1!Gf*wuQ05wOD}`D)Aimw8dSDyZhj7gFA!!VD+Za<Cc>&#ocunxH!e#9Ukv5?7ie$2k4QmQqk|wc^yZqN<D#P*f}L0T5LBVLy(6+E'
    '6Z7^CGZJ6YCo-&!kx&fluzPgDOY-'
    '=kbGaYCHqzC>zm204es#F;~w}w=akD$tCixJ;yXrf#NP#ma}!7$8;14U{|STkU%`h`g3<;Xi4H=Vp!xKDsAj;9HbNI+@6|HVbUHt5p-'
    '^HL2^cNBHlu-d0sNdl2<&b@nA-&S;`j+yvz-K1L(d!o?JW~9e1pQKlu!<c>#=yFy4*z{60S&DoFjm3;5)H8wlDNQ6J@-'
    'XpPHKaH%82q6%Tv<Qyyo;g>P#ML%Wr_g}tJQXbv(Q`-R(ySmXiJ18a-'
    '@iwBf{4HuQDzT?8cz#>0XGWIwy5*bdHB;VleML+S7dq}jEzT@4<n&eAjo^&VroT?Z6=KZA&4h_@GK-IJwkwu(U*%55vDG^vK9$>@5TF~'
    'x<#WdkIH2|x|?3R0{CTkx;358RE?dD;|b*g@<0$BkB7=kuLS!ag~6gq7G2fgjI_$l$6k**HIm(o|68pb>Qinx%uN7RO^pf5osc{qshi>'
    '%GTaYzi-0xv7w%|}BFzJhX))rk_$M_5DlKf9~2Hrx+-'
    '1`cx8f_eg*k<&;^^ck9<_SQRtn>0wPru9|UDIKM$YJcsNx?L~wPZFmUJZ5Y7GWmJpO8%A?;(IgmQsS|ra&;CW7DXFo+{fm_6@VMzxFWa'
    'hGV%^8Py7VakZ$Na^nb`~^l7M(=_}Ds=ZtuEAF^FM8;!-'
    'P*zyW)KpGl{#bbi{20KB<BIl6tXcH_+S|giWRq27@Q{fen`H_C%li{X@GyKemh1)`nfOQB8v;rSHrz5k05x}I3DmW1BZro>fC_$|q`2%'
    'h4CyCqOP+b99;t5DBYDKr8aWZR9f~zW%ea9N!G;-'
    '8oku*GuYA76v435S`dhV@Vn2UbphhgW?me4q?wQa3qvv+A;nFe{$O|DpR5J|&5=$Yc)$VEDa7*Y5@J_l|=T4H_S{k9T79OwbMqYcpdMl'
    'W>+Q86_Ms)*8jLP2MYvmN*Bu~7ImV4d%YqrLv2z-_C~$$$p|9DU?cXurs5p_jH#E^icTQ?>Q-'
    'B%@O_yXZY2fMJ{4yjAiz%lNNB$B>m^mg}LT9W>VX&S}IH0HvloOK>Tn99It)g0z7bAY+j=&K58QEwi<AUPm&mrqF*<4jBT%N=an^p9g9'
    '{vi3vzqFg|FW(^bvNezWHGA1UABSkYmi7G?y0d3**uv7Y;qY0nYkKW^Ys!C{ck$Y$pv=dqmehrL8dE^^16)6GD$80G?@?xS{cwl%kGmy'
    'EUrH2;crP1odU-'
    ')L+kJrY%*jda#D`Us8f3=tDHl`|fo$O68##QJAF%ca}3N)Uw)tw*wQq#dVX?VbQ%n|nYV&hGX)I~}UB_$XWoDcov_HlFJj?!!MVAN^La'
    'MTh8?dqHO(QW{3(R}ezd<=h9Je=Q8;+Zgrst=L3`X?!<PLx{7cfJ2}-'
    'f@<xSJEF3?GzTM?Q32P%FgYvy_g@`QBNglwXi#n<ez%KIbTaB?4`vQTmTr26a>Dr3#C?kcWI;lgDpXRfgFW8M{?CZgipc5v(ihbhh7O&'
    'Yjw7DLb$l6P&?=;*a3V9wGb}aD@bDyjA=u5GCqVXq46*qTCO)U&O%*v1RFwKM~cxDG>omra-eA-'
    'YW<5{8GTIsq6>(L#B^6J;u-Z=+(T^gh)DBo>3REnUo$R<Fqg)@%WF~)${4DCGU5<tVSBs}m5xl-PHD44pL1bxJ6|K9`8p>pOrGy(wS=?'
    'VlB<{pT7MBWiPElR;V;rXvJRCCr4#+Ix#$)DN^2S9Pv{){1<D7zK$GAjU^jFXwwPVZ*9&QkErc?X+(_T5ZJ>r^mQ>jYZ0Qkb<k)9hz}Z'
    '-*snFYx8>aR^Z*Y5k$DqXy0b8-J>2KxS-F%5C4zA;)q+gzh{v*8>E1)&RN2mu10GWE(@Ijzc$PG^-'
    'eMFqp0O?K*vXsD25>t>^q9M8h84I04vcO~7M?^-'
    'joLV3o{>Rl>+{X8pU`<#5&|vkPY?B$`fwV)MBhHe3Ntc8{(pSG#h$Y{VH1Vf#Ih2dFN5&Bw@N!tgl-'
    '91<=s`Q_d}AJCt7@+YR0onE3NjpRsK+(OhKnNCjQe`)to`CK{2$^kdAc|Y?q+XlegIx_Ja=vdB4ADXp<oP9g(0aAtQ?9r78(~^$(odMT'
    'Q1{O)bf$N&O&q;`V8@*{g77J4dfGQMPAEYpbTP?L0}DoI95c~uvbRQ5H+07kTG}*xB}cBDGOc&+8_;(lE`<crwVH2wUX*z#(v(fPf|h0'
    'TdbXDMqp6db8IvD7c>%^ik8Dp!gY|5*e|rcwo3g>bZY#|VFES*F98j#gzWm=jk^#q_kqo*$Jk;VTatGq9kFe+O6FJccs<WqJMg056;Q%'
    'gUs{F4^L+}oo37@Dq{qN=E}cJX@6Bg&vFs+cCU+?4ume(#)<r8#FQ&0zqlBmQO?HEFhw2YCha#{Unhg&X0`OXh26kifg`Y|pd2rg8-'
    'A{ZQkyhXvafQFJ^<F~LKn|C}4dR-'
    'x*@3)Z5|_<x=P;JVFWbjzLG_!Q8T^So0g#juD$Y=*c;+}d4y%vlVH#F|vDyftIr<t)*4u<iC?2h+T3>w(XCo8wmvCF4*b#)k6AQ#7axx'
    'l&RmLFfm7TYBL??*7A`+1h$^w_8*>D4`Eu#}lqo+f&T*>HtFbtv48pMaZM&<*l@FjSucUt%$vpe`V_nO?J)N-'
    'w5pBY0_M}{{?t|8~qiEwZ98vHl>9&CVi$GZV$V_N7Db-'
    '@s+SYi&gRqBbJpypytbeC%`SP|QW$Y6)xO<9c8hj+pipdG(|SvIq6;#Kq@v<TV+Y(Q@1>B%MRoumiid-DM{Y0p;wdh1TO9DDP3!=4wfi'
    'q$1B{C8-DzmdlNrjQfrKlUaeO3lKy5lykv*gO0s;zW+2Vf+ttBUXS6OD(^vivB}ArT5g<sppIsjus#Cr-AjgYTNfgLu`t>iWLs7Ogs<v'
    'a=lU;q<o{Q8!e??id}1|ewG$xX+(F!2F*btKqX`(B!d^=Ccs*_XKspdjXUQ$lyZdc<vu5d<;lL8VyehE_XeKqI&D80{OI2yloPtKy0C<'
    '+%iF2y_yzT$I75YmIYbxAg?=WoQ9K&Ix1-'
    'ID%=2Wxb&(IK0LR0n(I(I&eYScH^QUxRCn$&N*AX|!AH)oG7GGQK$CY9&f>R#KbqJhaeMi`n)LxuTw4kaXeWQ&Dg1m~}py%1A2SJ(Ang'
    'mAktK;k1I|EPch{t2EV=C<~@{GV92UK9Is{(sPZmqwQNqwNSQrw^(Q77Aa?rCo+2R1eTvYp>K6S!vk3~eALQ@zQtF1WB7Q>Y)JT51U)N'
    'nZ;;Y`~f7m_PcZU>Eoud?|3+@yJn}Jj+u~>&)SdV7v-'
    'F4z1B<s&(|qnu*5{0Ir#k1>KVV2h{~)!T;FGKt;z8{vYZ})UJOKuISUCUBqCl1Jo+8fqw<82aX``g$39FbT~W*J`Ju`3%L5keRx(h8s4'
    '+}YeZ8|;XX|hGR-l-V(ez@GTTA0v17BPn|YYMw&g$DNC$}oH_dWYVY(PA!-urk$T#yXJx=?g-'
    'gk6?rtrf78ve&U8@TO@0bfAdY{y7VHb>?fu~9QSC}SVR5MPApvPm}^>88iH1BiuB18>2qNNuzQ)DK8kE>ihSa_S@@0s9)fhD;~EBF`X`'
    'YzbTjs-RV&zQ{pfZ*p}gXd9AP$?Vq_iYE$xq)dR`nLhFf{x8n$ah=uGa!5Wdm2?l!Hv!WFU--qsMeei|=l)7vC({LI!O6g~oP5WZg0nk'
    'MC(6zzfy>-0+kt|`Tpyrn-'
    '5Bp(ZluQ{wX%Q8@j`<d#_Qu8kHDA+4Vu+(s2_JfJWi4vqqm&`{sXVFm7VK=l>iOSv$rGMRBlQLXouYrE%G$q5B;+INn60%qxGcIfmeas'
    'Y#X=TKURDoRA;C16PdGo#KqXViC4s5QZ%0ao$}ohR|DBXBXfm78NN2Eh>fg$q+MB~xF60WVLDt-TAD&7Jqfn^XCH9H1N%(~BQ0YSWD{7'
    '-4i#FmJy?^!kiE^vDXYn2@+`}7sjsP`93bYC&5$VMLm{$ebQ|7NEE9B)JBbd$S`o#|5s$okmEyv+gb$_oD?oW@0%G#_^X`D_L+#Lyfpf'
    'r5I6=H@o9yS{8Ql4#TTl&|m4*tv=;r>Q`W3FP&6audWVuPGoce%=`7Z(yKmwD!qq)C*JACE1dqga;m>Op6Q5`Ak=+5CeRD*Dl93a1O+p'
    'K-ichCvPp11|TK^Sj16iBwtQyFYLv{JYvmP<H-&Pjya31V4Z5N{%ny@Ms%Ts66*ryuY)6q3ARFegu1E)7C6@)gf7qpu+uvhkN~X*io-'
    'SMU&UA})Vb+YGNH%_*20_-XHDT|>q&Zwxe?Aa#Wsi?gkrf)QyMQBN}Hl6pxNA=Y87(PQ9BuspI0n2}ZKw=dsEYlR+pI{NMUVQnqGl-vt'
    'fNI|1c{3(9+_uk6I{zj{#(=Y`cOPu0{8V`tvN)s&xZ*H`X=mnF#J=8JIA~ge?rd<uR#irwb7Gx0$@GgIuc8}*NvwBzy*ePNqScLW@RwH'
    'ZB2k7*`So8siplhL{s0o$f8|WjXF_^12)DV5Cd`@2xf#`d>?6auL(V$U44hGHe256)6AoM@Sd}uA|Ho1Um1Y$@j9o3+#NWCk!K{rR<x%'
    '|=Z{Gv!2G7K$=EF>2q|ASiKZ`382316v}H?E+g=~l`f<$=^d-'
    'lvF4O>MSfW3SuFnNFFjCbS1WS(6;)t$)Iw5Y9D9+Af?)n@+p|_Ev0Uf9rUJT+gcr_CY%O@3Ra271#m62mWyda{{FU&jXS=!I2t%9ucS^'
    '{>9FUY5|EOFZp~eH@{Q@=P=nC=3ewIu<ePh4c#iZixjv4Vz=BHXp1<E7HUAPtqYPlv^=^@x0*89YG#`_-j<vf>**Y-'
    'F4ne9p{C<`#0M~mK7hUkM`3r=OND#Yy^)XVSEMO+6os%W*d%;3-'
    'T{7yWMD^(#&QNZ(U+XEJsJh?yJfzCd8yV7ZiNQ)$yfrh2I+xRK#CC(Y=LZWlu;eoO9^kvO74v~(h<wXij>x6m6vPB{^!H|Qeck|=REGY'
    '>6=pLssDrQCtH$<Y`of`poV=F9^;RJSL1;c#IU65=qDtBSdMSU{OE9O>u+7(#5yST!e-'
    '%ka#bP{tpNdk9nQzELS&RKY)8DuPf9>Q692)9@c!g=$2Fm!@}clGngfqDcO~`TgnA*;l>XC6FlG_~oiL8vK+2)jkumTAco;Grc&!v0*O'
    '(2i^(w{2sY%Lf@te|CI-'
    '<1}B7ts!svcJY7B@5R4__U2G~31}P@nCMnU@)5lnMAetQxSLxe88Ui)4naO}^twQ*|*LvL3VhH)3?=0o0H8(JmoV;tR0Lq9~jb`w&&>H'
    'p)7=IniHjhu4K`i@*G1<wiL->5IfVeWx%B<+UStA#n>S3*1D%h_@UJmq|EPkVsLQ+jgrZ0SsS+cO%+j-'
    ';rsAPczB?ljf>pq^9~6@Pm}8o=S`p-hv^d1EFITn6qF)E+KyjTqZJYZHN~93H5|y9kvdM(Uw3_wTG*r&bYR^+-'
    'apNwxly1ld#H0vInAvDw}OH=%#o_#Yz30SF!LygTGwuwHW^c@t=ae>KE69NDIBUNtSHXyI@%_pgx6K(Wh+-luW&C-'
    'W&F*vW)6N&tdA&J>+O&IUtJu;i@B$<P~*!C@(aZYNS3cI8|_+E`?Ph7Kfj+<80fF(Yssbk8<7dN2E1re-zlXsv*Et!RABi(o=|va8O?r'
    'B4Rn}Pqlg|BjBOir&QPe@eU+b5~cU-B|qhMCJ=Oiwv}rDj#5UbuQemGPoCA}6|+Y|LI=@NTCx5T{X~2L-'
    'pKJ;!;+Tp9aOjeFK;)N0IQIDYBqjD@#qm-22)kL#9dT>i9?B@^-'
    'nWD)BbRO1Zp~2x~03l=NsOY9gkc$%7!{?x9}Tizrw18#gUEbD93Hpp~2h>>5jCKt**8a`#SLOL%a)yD!aMEcw6eOK1x@F+O`G=Mp=_MQ'
    '`>+h&Gb#c6WLpIIsXX5sg@AC$>%~2?G{>69_o+-'
    'E21vEU8FztA2chWz}P93aUN&hN<p@)W4cplI3+Y)d=feZ3?w%(R}hZ`+p80ul|#h;;4>))f=wwHzi#FE6mfOR7c#aWCSun65G6npdM(&'
    'rJLAU~UfF;x_j@Dj<=M>WgctsWY8Ux4kN~tq^vp%#1hF3Yis=g`6A@6$zMa#J8Uoy7-'
    'U&fAw7(BfiBCW~1^&_I8Ec`s6dm`KZKmPqJ#w(PPOYVprglPH2Bmawytbf&x<h@~;2<$8$q^W(&NoiM0XqP!=dKF%=~<AV?~wn+hbx2w'
    'K^?y1TB>s>cb+{?weYXb*g!SNP1Gsjj&!?%$JS8)$F|6pjn9N?5yjdHtOUHCsE3r(187-'
    '2hsvV7!P&A6=@=+NOS>j=b&$jMmi|oZCE^dY95J80?K!8W6Wb}T@)z($c(0{X*K-'
    'Fl>sb!_*H|naZCp<7=DLp;H4z$JVJ=}akSSpk2$EE^Co!<;8+q*So!cMz7@-'
    'Z^RWI}i#Ec?KQ{MsC@$dncsuhy9ltm$2Q^d~HYj~d4g8V?8vFy+eN%43lxiIA?xq~Q`kK=RbA#zu3l~6gn951IOy52Cu!85+9sSU!A@+'
    'Djrj1vjqvwI<JNo&uisAv53*gf_>#0GMlHc>91zl-mQ6m6J%-q+n*%U{Z1U0+gC2@~HJT(4E49y@l?WAHvik7yA;39QY04qMr&Y)rI-s'
    'HVQdEBNjkc5g`QE!A|D@GeUp%AE`j&A*`L!7J5@=zJVtWbbP2475;h!;HZa#3ksj?5;#r<tDS`ce6etPhzvnztJN_PcfTbl{7SqT2Hw_'
    ')Gd0Y?_jTJySNiXb0sZs)>tYov2TK!FyDRel&aiU_QdbaIf<NZG$Ha;jT~T{Mw=iN^$o=7gjjumU7-'
    ')5pAzN~eaL8VRbYs^hd2*@k}YJh@2U7rc*di0jmCg<Tz!^lLr-~+X_9}HTVb+-'
    'd%Q1@(m?r~JbP8KtF+WQ33EyJgc1ZI6$D2icJUcHowhgzpv|!1(j6FKzT%D9(b?^_P3n_CyWntq12IHvOwL9&p>r)|gCfxZc@4C~YdXe'
    'Y7*r+*qwLQ6fhoibwN6lVc2vjs-'
    'l@ZbmxQIPDEtNYl@1~p`2P2W4u_72kA8Q6ReAs|20oxRS2{XfxJ0GXFx6IF!#spmQ1L~r?Q4vXQbzj(($Yrm9KFasRhkK?zpuk%?TM7='
    'C1k7e()R#(%&(x9sxN_;NN0T|S`=AfzOSs)z9OP@5lmn<iggorrhKOUfJgI3iPM&+;t$d%|1I^Bc9Ao*UF3+u$?7vMz#O3r(-'
    'Cl);RH&McY!7ekou`0jv}B~V<LUXWmfXTKBb|!8hQp@mdjZm)BUOMOf~HS-'
    'l37m_1KpT_|(SeamMf2X#6YMjELuH+CBHMa6f$$`3S=CmB~NU=Ng0VcihyLIIG%HoU@thhL)SI>e_nZoU;)(C|DsnknJIF#%WJ`ZZ?*U'
    '4fA$3j;a8ao7aQyD0Waj!6(#C_-py5vIMPet3uS0D<M8{HBrmbojn>lpiBzH6g(n^f(Q7<dM{2U8X!wTLzo=QAmpOvWLxS1Y?haX=}0q'
    'dsn>@z(ZSdh&7)cxZ&lm56N0^6)8Mj{lyHmO6qyp5EiN?%>}=b3n=#mEZy*G%9hUmKVu_cIYietUfR1%_A|j|s8Uf@JSJYzl4w0!u^_T'
    'YRG(|K9pDWMM7-|y5rL}5&MKb5LO*r5>$i!$Svb1Uu?&_5a=L8>w9}ohQ?VYOP;c*s=wgB4bmkZ|5Y=R;-'
    'B>LDhbRI01_7X6KMl##*B&>2+ld2hyoem;P@iS=3LzklTRbQcJ`5stPZ!xk@dxn?csz-'
    'Y&9~`B$M)U<{Cuk|MOE(1*<0464Ayp5&YQJPVptkXS5?kW)_@U$*{e;u*xFFs~MyXDI0D6U8Nsd&PWUlA-'
    'T51sEkpBpS>?)4**QToA?Z9>pSU(N-^u=R#LARKs1@Jy0f)18GNwYlxOBUJOn(Lk=H}O@38gi%1V;%d99AXk|R*pe;_)$z!XaL`Zk-'
    '6JgQo%#|rIu%ZjbHFI7X~D_%v;!}_N1iFLARrfx0~rIH$K>qoo%bSvk@BwrYB$Uug<N-Rq~bu=7>-'
    'HMgG|2+1#&SL;rT~Vedb6r<v1(%YsjWyDZOb&pC7OqCK76VmscjhCkqM;mc;{aJ#_qJS^;BF`tlFTiE0u7;Ix#frZw6fm*<m;AHDKAs@'
    'KR5Aob*CnqZn>w`b}Leq=jiePR-(3{AO6+zEGmVhH{O$*)$q;XXony{;&fGx+1xd#NwnDLv$tV|W@2R>OBvFuP=sQkX=Oo-SP`k06_Ww'
    '6=SZpI#MywVU&NGXfj{g3TIb}Q8(<r2IRYb%F{55!p|O<krB6*mSuh$*fT)YQap*jw&x%6YO?YBE)Up2H46$xs08V>HKZE3KsAnww;e2'
    'gU$?TjT(d4y1>i+!Nur+||^StlzjRHxyq(mjOCxgTh0Yb4CKPhkliktec1}mYh&G=>+Z$%goZ0QOtB|im;g{g!F<N+^@vRK`Zyg7R}C%'
    'D{a5PRq`L>=2@FKiXF}QAAz^FQepx>&wWG)dnQ|M1GNKHxznaW_CNU(?!)|IK66Lyq<hJi*-2b$+dg+c?_a@U$1tuJHyQZI#yB<vCi-'
    'u)2W$?zC)@4s$L-'
    '<yV0O}*!1&tc<OZI8Tu=8)_M~s4yR#|TiSPXB`^=BzOE?BtJ$$3>5XUwc=bvTYVQ*&kTOU~tC06I(aP7QXY$5hUT(%{;t~5}FeP)w_Lj'
    'lGAC~GfUC$UO!hqHPxfvwAixT^VG-'
    'M`p__B#!I!Z<$5?F#nDiu0hpKK|^2*6c&>qi<`{MN@O%Ie!OR75fD4B6}&RzBN&(!84xrKt9)=#n~_H5pKMF1lWu3;(y0a$!^7^dNJQ3'
    'U$!aUf2zQ4f8rmPc&xycwZ{3t)+{g={G2ev2ZH%N#C(IVVt?ki#MR{Qd)u(QI~tejbD8sffBH$TzB8F`N^5dEEuBapnltC43m_lamYs)'
    'OmQ%@+1haRav|l+)oD+`-8!_2l3pu)H4H^iN_+5tu+5S6k((7~x?8WEH3xG-dWq7%MRk~#P1bvDo+4f~LHhA=svktidYXdE0FJmrsB7R'
    'dmfPIt3%74m|X{L5qU*Y~Be<46_y!^%YKYFh))&i-'
    '`)Hm!gDPH5;d67y+Ke_}oHmSyMMou@nhS!@$Q60#Yq@+BcKzs)pmgaFQa<AC8`mL5p36F#&{)53~+)XP9RQr2!lE~Nd7PyD8FB)hG-'
    '3!hqj4|JJu5%x&f2&|~(vIIBti}xumKPV>F!qNpl2>FOV%4~h_5|^5d?xtZF+1R7|Fk}F(p+VKYOrjhW(6B|P7NlAGucA6D_gFik00;9'
    '&bzq^T)bamyCsM0du;QqJpy?i)!B>h;f!!3_bR6fyDavYbFBFy*pwU39`scRqAVNOZGX%Da2!i&lF(M#Z0{g>15HD(VN=wE4PqqXiP08'
    'mOf;v<q@_vANT_gM_&uHKYXeQDyBNKrnQSffWWhu{;L0>rqE$5yN(n_nI(XX8=d7@;2p6%7Z3)h{@GzTEG+lg*v!R(tDNbSjU^3;>;Cf'
    '?)ZwoUTsf&%0fA1}HzrBiHFT78=!dE3)QS%e~!et}FeCgOr*Cv0(gad{lR>FQon=$icjVT{quPr6E8#U;kp7P?`yvn{=%se}*Jc7u$<)'
    'oXuXS5?ari{qhK&OFz>KDvJuA7`pJ*=E|0@<m3R5`IZzf#R%V{IEPv9_%g5K8j8lkcD<JxAQv@Ga_d`3Zi?cUC`bh~9kuo4PZ6A1de3r'
    'K#Y-q*`=+{wGsgxvWkQtf)xUkQap}d9twQ#zqLrz}Yr||G5}$i2bzGKwr$N$YdnlH$-'
    'kCKN1e1r857N)38{rG`m#pZa29uvc=*Q;s)Tu<_1{c25>_hsGK2F=u*OVYNWx3K>0BG!q<<Yv9{6;v>0Bg_oXWe4q-'
    'i6%!IjI@6Z6BzZb8QbwFzmpXOm=x_5;!7TSu>P@dQ(sV>738RdKHZ=P{IbP3e-Xmm#?6X;5R#@6E=-'
    'x%92_lWRW=NU3cTmUyHRf((cTQ!!0{4{xrsKXv(qv2KVgN|KtoGB#bMvl_Q5tC4l92nj~EeN$VBHp**%zs8<Txc%907b-}z&xG-'
    'K~KY77kL<%&fc;$3Jh~6B{PAy!8Snk;NG1x+2Y)R);$HS*k+EfyA3qc+ZrruyWp#2)A9kXmZ`tvVWWZ0ET70K+&t(7>jB5|d)B{WJ*Ml'
    'xpNY5JmBnW4v;Yp|aT`qqjxF_;u(JcEz(n({f+T+z`+0NeWP5O=Ev3=H;98H)?#pdtLAfKrR$#Z-'
    'Bktp#*xZuN#s1!2$ljMDJ=@)v^GaC<aRGNm00gfl8o?3vI`V(iRHzxem>!k-'
    'C2S_o&~&ajx`mpm9*krms=ZypL2b9#g{c*4DgOXFlBXP{)e-h=bOJO*RL~+LpLmU469#cxd;`M{rLlh)(-'
    'Jsn+Gd<n#xv82u4*bBRxuw=FJ!JTR>d&3YSZPAUZ0t#wlI!T9mV_Fpe(;WEi>Q$LVXuZhBr`nUJWsyI11L#&xDqwC|HbCK%a!(Y9o*j;'
    '!6HJx&yx;!^$9f2I|I8<RjisZw##Vd~x)S4oCYTH&oI2CATS6FX&0FV!figBPXy8!BSLHaVGVRJZ~>@jNyOms!EXfmW&X&L)wVm)Hca;'
    '?L%}*+a!+B?^5Zi2iOVDWo7IuQCY3W-'
    '_VvOZ@{Nu9WgUe<eSHzXNw)b{7<3I+Ccbe94tRV|HYREj;L<Z!~IfPis|mUmSksFlMUadox(dJsqR<!8)X75dSjG~>~Z2oq5@c@uq6+|'
    'W^xtsFZM|8#(U*I(c8uW1hiQTpZYs%LEwOAOlpC!%otg4jA}`qAS$p6eP<kHkoQ5-`ox~-'
    'bAs!|>Bc>d^SJE&g+oFKb*7}jN9Dmp50Y}WQrBjFF}~}owAB$@+ZBAQ_g7Z(r>q8f0{}v{{5bUz(uLrZV)Gbv3c1-'
    'dwxE&FBOI4g*)&(E92mpC6xUh}DT;p#p65Jhp3x-IhPp-eg8g_>;uuW><|)&qAMmT)E;&yDQ~q!~W&ExSP$_m*SkQ+P>A^ZoCput^<u{'
    'XMu}i=gte;@kDZC{)I`qc3+yXoEiOY;vzQWJc;+d}GEwNg?ipHUEDYO`Sg6xh~4DiY)eg#|)sZUE}ZK<7oVnKnJMD|k7q7GpSzasyCj-'
    '~4E-`jiv*n@aaoYnwfj})VvvKGtxy+gRt?n)_Cu#pnU_!sR;cFnF#)WCc3l=wWDhWBEM;ga$(lZ^cZO;@_0cEU#b(OUj-'
    '0edJ3Niz;Rm*ra2uc9Y1FbGpc;ubw3epF%Y6gFF{L0yZu0@KhZ;O=lxxIiZXN4+F=LVu_`9Uav0zWv5?`FY`3=8iju-$xINYyq#-'
    'Tu!VlEM}`lriep`gW%M-F<7?z$k~V8KuqPfMEXZ6DZ{uU$RU;$Z#g@1e-TdpSuhu@h+F~h;ywMB!;<uv9citYwpWJnhiV4@S-'
    'ha#pt6h}`1V4Y^60$yjPA3owd=ueVz?VBLP>CP&SmW<*$p}zsziKga$R|9loKcEKh%Jyu`6B5*$#%^1?^HP=6$5MUJ@*&#`wnsZ|i>)<'
    'O4ZGX`rE#NW17CWGfvh=L%EbX~DIU-2gPAKY}5!o>4&DG|H<@jZ0K{HJcKBfAiBx4jC5d&z^Q1pw0j(jLlV@o1}-'
    ')>p*GJMlj5n&?@|($%Zu5F7kw$O*F;*<QWyi!&}lY4jL$LA%40Vhvw{i9_|Nyk`|y^XfAhHxf0kRHF0GqJtOrC%c8HPaaeC)hjvX_!uI'
    '6aIy!@dRMvQa)DRX)8?X`fLh(H`IAe!@KRJ#a>q~Gg2$V-'
    '1uHn9O<U!0uY*p{nU8rqW`k*fIZ)^bBAbO7oNM2<Pv?{btk0|A+F?f4>s*;Sp07npq>sMpT;~TV9!Ovh2&J$OuAu&ywKsI$FOx-'
    '9e%31=Kh7Ls=5v_y?_GS7X*qKmXIOx*Bq})@WDQyIm7OC{xTetAlDQIZ25zw)m+SGaBc;Y3jH+4@r>E7f%Yl?{1m|cVrN#N=+ujzYg6J'
    '#B6Q!*?6YORfn>K5lh{l3lXKN>tyv<+)uB!Y2NYpM)z1?_}h76$<Ji2%Nb%aA(C?;U=zNqjTW>!bLU{55c!^Ht=N`5v^*aBoiX{}47KA'
    'Dp9%mEsw681q`}XLKOc@Nmj(O;;U!8=^UtlwI3uWnL(zPy)Uq@*FsCpvFvYzcMO(M0=2W1N|hwP;1dW>??`2_`k-q!fEvJoDN7q)-'
    '_`z^#Wg_bvC|>iN<0n8GRf5Q{1RGRNIQ<T^??Qx`_CK4pmk%6^%8%Y;Z*6rHJ7QNh)`^y5VNT8Tok1JEW$Zf}PW6F<oez`K9~=Zj<*M9'
    'Y$ul`zm`u02~H=hYRr68g2fdzM)!xMND;hxdR09L&xN$(!WAbD<M5bzlhJp2E=?ciNOo|qcD2~>*3E#Zld|ECyiO^-'
    'Eg1a0`)1en|dmjQu~uffC>fU#hl!aMER6vTqj0!Je5~tkF5=PM}C54q=qdYf&2&_80K#oX{-'
    'f_(#j0^Ltu_2LW;=j<ajL?^UMG8M`WweQ|`gd=2>ATwuppO7}$U~5R>y$(gHEl*Pa`uPGl>Ds*o?~<Ip|4wzMK8%uNWJ!Rv4~GM?%rS?'
    'FVUx-`Af52B`4g=it>;-'
    'rIfeWz+NDd;=hP<f__{482VNNW5wB^b()B97(Zdq_9=nCP|tEer>Gl7oS|^ltwr6;a30XY$VKRc%YjN%f9sx9KB#8Rt{JuC$#<#YWJZT'
    '{X2z+#~1{brO#=X5xuT4Wh3$INY;vh?2;Z4o@WB=+DDD?RrkTWQMOSc4LNv23Vg+4IWLJA-'
    '|$3ut!3YYYzsxCg?3gLv;{(M?#_JNER~|4EY779$+w)H9h>u@s)u1>!xc)&oHZjp#hQBOeyFfV=_w9pQXW(5Y)mj<ZQl_tvOlXt`wa^m'
    'k5t=u9D-EI^oHMpO6<u{b>EYwbTyoyUd4riCQfxH+ng|0|1JitIycCMl1JHq8KgmmBOp-t$@`a`{V7DI#C;(P<V?v20pT8L-W-GoRgRr'
    '=4>b2GhL;e)qP)y3x)NR*5Mh(HMl9l(B@DSs2$2fj{@I|Zc%59_ta)(9qrX<r9K+pa5sD;4~HIm1|a99RgGHd7hSo@K37l8V><y-'
    'P=c^qjT&zp8o1nBfs>Uyuom)8KIJ%LS&OG7q!N4hOI&%hK=$xgi3z#YM4cMWW$9M^toX=R+FzOd<oyu%-K1;-'
    'qL#`6^AXE$LU+pqx|c1()WEjKos~d@a=bxaI}h0p11+WhPoW*iFD0HB4^2~+sR5)Xao63ETf|lHmLVz%Td20^VDgIIjBV=L;7*qI1oyB'
    'X(3j#ev5+#@GNClNbs>ozO=ZxoL|wG~Z{=^*!=b*|LI){z4W#LqWjggk#cV&(76f5CEV{5nvWIa3-'
    'O0QlUnL&%T!hQ}Z{f|zy52=dz!TLAeiQr^pcw70mM1^zT|<u0G`ftm1+8c_5hLsdPo7?rXx?y-'
    'm_}Rk?<h^R$BtG>n_UZ)61F%5wRnT1>w~n)ln~6LG2x0l2Tdg>Ldl`2LR;jsxS#&%{X#a>r#La=mOe~GFd;I|*0V`xX?rwFLe%P1zS15'
    'JbBlu8HBTb|$O%kP5Bg-SFXb_wJ2yJ|vn7yipn)uMe9|t39y+(VYZoTwlfnx5Gd?7|Gcq8dIrxRx!nUW5Mm~yWIh_#k)_6<zgOG)u9=@'
    'dSMBu<vc^=swtbi_$EOW;P%ZImkN&|lWmSh$(!>ch#9mI5Z9zw6c_vl_>PJF@4(<+BPVxOo3QZ%d*4iJPEM!$jc8A|&ctZr;4|5Tec#-'
    '%iUD44|;22Uzh<g47CUPf&6-$(Zc>Z7PWON*t`m0yttMtgA!RzuaJz44juiO~bXRezE?Ol=Vv$Gvit)5;JGI}@KNWyP;hyYTJ77br-'
    '{k)?2}@I+}Y(I&iK%MJ`qTH;C}o^abWS7f=r39u;S<i1mpXlbo^V6u2CrzepYDq-(WK1Ggu&y#Z-'
    'k;b=(MbK?q!gZAM)W29n{i65Zi=z)s%c<jm`mSBTA>(S=P*+LnnL1IW#UTC;F%eD`P`xIDbU{0zd?7yM&?42hg(~r##rx<X?Hg5tT@Sl'
    '8Bl;M)Cs!B3$|Y@yI+#7h_ObVIOj2iJ3*pN08on2DUZEnB$x-B3@-($f>i#>Yo#1=6P0Aqu`+A$P*Ew_b<NQNMF7jTgrjOLl@JVES!-'
    'hMfKH^MHOKl^uRQ(Gd7zwG>jD<uqvWu{gYlK<}CcIDmB6W|yC^^Ld*HxLHx{042F0V3t8SENz8kz-'
    'IgS`lUcCN3ZF9u~1Yfd?=bihj;78@u4b&_92b+*(}8fc;g;S?FtZU%Gs-oybhV1ikiSf#G{o$#OJHs;F0Z1NJ}MQVi2N(~_lW(CLN&l^'
    'm3)!>hEo0EEJH{4eBnH=$apg)N}!R8JR`;mGZX>YqB+zxI-'
    '`?8~j=B6b65e$p%nM|1EFOn;<{lsImJ^j!#8yRQ$f_8v9^PPawMoDRV{9R(OdK)dx&h;OJmg4E=v%y#=3{+x6L|egu-'
    '$u&F_e?F+r@<37)~YDc06GB%HGz$aF~%-56&cR7MSpl7ast)-'
    'H*xclPFCTb10NhQ>W_k7{!?gMWd`v%G*Rv2`;Qp&&mY+zMNE8@l;)^F{?Gp$iq-OoPDp*nHn1!?P&ysz0&wU%U}kdE7lht%CzaA!WylP'
    '@ud9x=CAkvloYG5b1g*x`<BP&RW)rxgkVY#1?u&Np-'
    '|AN4qBH?tiC;nx%WUpGkP2GS>WUo%*|%6%@~ITBwO2>0W)i^Nu!0=o40lDP0(mH9HuC_vWciVSXw9Oq>n~v_0rNh2rQW;T9BH$*NZb;6'
    'h>2KHi1AMXI|Z6?Tg5v}GcFmNitn&lBXbJdxyJLO$Tg_I#6@orgCq%cqN|u#wT<gkN`xYU<H@x-Gb8)-'
    'Jf#!YjT#SIfhF=N@B$ITrx4k$3`;q02R(^=99V*o!DQ-=auInKx6bsa;BO>eorsdk2bHG)8Z<TVUDNQys4+y`>DtTlb|-'
    'RLU4|kd)qj^^Wz>Hw$rJh^KHy4=R|<y7N$e_hAy*5q6SKI((Ie(@Vg>&tvNZn}I7a)Ayya^ntq!YP9OGczgeQ>g?H|4X+(zahw}mC#Nx'
    'dmGL@dh;2t5`3xr3-'
    'b4M*BY|7Sck#q0X++bwJoO8av4VN5<53@h3NsB>Zig)5MW(&5T<Md_h_&6n=bI1gh=+ZvHWA63GBjoPYCQ1^s;usxW=`VLpELJ>OYxG#'
    'E*>Ev*_p8ctM%&9mCy(4onR0K>@Q`FJY6sT0;q{xh2bDRrF4mzrLHxF=i-TBTlQlIJaWv&Z8#fJpe0t1cx^l@Z1-'
    '<5gB#7Q@~g=%+dXj-aJ&giOlV!AMM@M`2Od2GrC=^yQ@ShHwU-O8kd#}fzftHf}jRWugP0?#^r=|S?Hb2<1wvLWxY@8%W=>-'
    '`CEJ8_8mQ9g<`<?mt&0Eo@?^gt~0B@odGZI8Pqa$oo#@y7X9-l-=e^|<<31Epr6e#%LGJNg~@i`e1^h!NJ?*ctr^-'
    'b(ljs_LtOQdB$N7rCwem~Thj)cd=yf^pgi0drOhq{+qLalTqG1Ka>58+Y-a;wp_6w-9B3zo~ZkZRQL=-'
    '=z{Qqmyt6BaM~XcxYnbC*Sn&VR<_@i@f1I3RTt2JErsNh;!U8b{!Wo??YPa>!5d1oVuEAX1lK@;&XtP<bJ%m-'
    'T_NDt^w_XJ$x<cdrI!kLwX{5DPU%LVM76+U}NE6u#)aW7NZlC9cn-'
    '1L*mS&k+K}f@qeK@+kO7INL1hLbE5y#)`6wvwnlS&n)XXQPu>iPK9TE}lZw@Hy2J(8JlwBrmX~48fHH*DaVp%Q{%Bw?GJ+$~BlH}h#BY'
    'x85qR#3Cr_)b)QiM_$pdXJWuxAN->;M^sI@6<E!I+%Rrr?hUa34&i*^gM@%{2*q`&q_{Sbban&|u(>P+0mKkD@gtA%})kAiOE72A-'
    'wjugn7fw4t(@LtZNuKC!3a2nD?C+PD?t>9dK1^<^d1E)fx94q*7&J_uRD7RLr(FAQSQCGR4`s;mQ9KvG%dbKBT5xXJrN)xO%F+<D~OE<'
    'ZPCjTb?pxi#xo0x*Mz&ar_oV~@$p^}>J8bx$rMPKEVzr}gPXXhO8d(a=A?GTZVTHnCRq%y=Ru`SZi^S8R$TqxIL?h!3jHr&Z_5a}GYP@'
    'iob$u*4A7n5HCe-'
    'b(_tr8#l1M)UBIa~rvOVQAXElV$g*X#S$fz~QWKL)V<0Uu)`0gIM`bmEv$hs2vqb8kNWh!7Dnd_4LlYbX0Z^CEP#xXaa?oGPDI)0MZu<'
    'z7^rN#28CEFx`?ZFo0(H91|YNXA8bE2-GC?4%G@a1Z1?tY?MR7qnS#suOK9WIMmt+>WR$4vqY&%#+^coyn;Ld$eN$P4~!c3iSo5@-'
    'D|JWE8(NG!x5UN1+b1=AQA=aZ?~L3+#&!;TAAzYbI?s3dE|$C}juhq<TT=`BSN(;nu$cYY5bg>nUOpr)!?X@b?u=|EzRXDMxF{AlHVr3'
    'CFB6m?ua%A~idje*m9_X-G4nq-'
    'I8U;VyCn&2smpDD=3fJ?tZ&(S4)_=y$TRJXy%0Mk7lAmcN?Q3mA&5bpC+;7o5WV01vP|Lv`_5{7$Txj3JJPtx1XYg~mK}4HJ(Z!)l>oP'
    'Fbi7T2aI_FV;<m9VrHk7KUpPP0&%1(fZIm=ZtQ)#Wg1&+Z<K&UJM}y<a~9tW2}KNx%677kSrzNrmQ3r9q;HNUdLzR%atcsJ`Ia+l^AV{'
    '+Ajrx09SqLm#a5FKX)9pitZFjAgB2+6iqa`y5`tshi$}sdWJ#A98Wn0GybodjjamQNKqrr8+sEPkVBbxwLtwdjJiSet&>yFgKcd{PA_p'
    'j*TJ<U^(^vCeB#ekTOotRDms9@(W_9V&}v|O^sx+RCFnlJ5ZI^Y#SaJnwXimaw77BWU<Dx}rUNn5F5hNiF}Z|mPk&FHi@etN=`-'
    'QK;T$6v*(p~~lhuGkAOX~@Sc0M>j%@y$hPr}%@w;+kI)qGO?#ovcG^~a88Fpr+SlgRV%^{Fb)VV_QI%3FWm^n}qaw|s@hKnclBXD<P20'
    'nq#FzS#`ipm3vxf)s@ZaC83=Tj>ypX;ls`6+Mwk}N4!v9#|3!r&dl^@I(sVLE`<2!9kQ7G<^*!3G_XZNflrDpB5*=$T4(P#WUxNGkl3o'
    '~%!&J4O$S{g~~JKE!Kbg|;z%8#s+AR*tCFw0c@jN)Z}x4Z!Zu%b9m@&BAr4#cV0+6bhzYW9o`?xj{(FaCu94=2fEQw=!P%h-'
    '`+MBPDruhvtxTTrPF0aZK%ORL*<I5^0!gt!)g}!RUa;5DQap{O01hZVp$K#mKU7B9^cJAa-is(W%iHY8kq^qchQ0%rMGv|42Tyr0o&iE'
    '3J%QCN_j!tX=x<ed($O_0@aQzd|dagW=MK>`1mxA*Y2}HExhH+dnRu-'
    '1!H$!~TE`I_mOgpiHhN9FyeDNHpED?e_V+>%EVHf3JU@y;qRgiGgnX6MMP9PVOY%#gq{DjYS=;xm^BN>{wq{_i6iKuBR_8<6*GAvy|;_'
    'u)ZTzoLIfK=|`X@v?g&xAV1#cO>ms%rw3n(wS^15UcM@9qI(HP1Z%T5_~7n~&JMt&+=V&f?txq-pqHhAIm?~nZwjK$2f>z>?zVmA<X}v'
    '&9S5>7J4M_nY)Z1jw#0Az80T22ST{M!%Z>egA~183d<a!mt5K7YVN45OX#*~-'
    'gh+)JW}>BgK|9w>xK&XY?5_^6y&w>>knNaKJ^USzjGd8k$TWQpb(k3wK0_nXzu50#Onaq{Bi54_@Gjb&dYe;Uxn?8H)qL$2zmHyo%(3T'
    '($Eov~K^B>=7?sqcIbY}pt`$HHArGCOd<agp#`q%*@3AMTC{`HlNhQ+|JtKNbu~RR^s=in1jBs(NL%mjXUichgV*04>r9N0+Rf$CQI>C'
    'wPW%-'
    'quh^@lcL^6COz)kSGkPS|eI%$thMVXwft2p1BudIY=tWBth=uiKJ6(Oa8{a7Q?nf;5KMpOw|*c;B{cwf4yr?K`6J@cC{dQx7Zot}%_K^'
    '<ayd89}N3d3)0H!7bJkJ~ruS>kiQVw4Px4HhR%N$F{OEsY^|2aDBKP)ozTYo}C^X%4*+D}=JR9iF)8i*Pzn%%r6h8(X}+bWYjBeCChwD'
    '~xbsM131pm9OYgc8>ba_Me2(b?9QpLSI7I5B??Q#*dc^;NGkN{SkZ^tjEm41L7v)a^L{65luHf<SIE-IF|vzH*7oZstHym{C3h1BmudA'
    '9kfe)(cUu=NEC#R>IOKhzx<s=8CbE?85(JANIw?7H<+Nk4{WvHVvG2pz$&$bUSOXeIqNz{sKiY2&%y)Dhsb(+bL5k@QYwM?#7tW|qMSV'
    ';+(zyhLR}RQ2mW8a!nB9}NT~yEMcD8G|1z~!q_@zrsRcU&#upBw-pk6Seq<K9nu{pi1NDQApzn@0?lwSe^%XfsU9MIjUSb{9FgBRN3uv'
    '|#v0m<lEC4)271uk_!tda3t7H5XjBZ$NC{ui;T}BfkO@(oR>yhSQGx56q!{%3dk~O`Pjf~XG{-r=Gpi1r0g~|Jq@bZNz(B$YERVi$b-'
    'V+yS$A~;`x!%-'
    'ui)cnqMJupPiT{)mQjvBPOICV`e+qTfOy6)f%Cp8FX`R_`U_&9^au`qG4}ysLQG);>p~!UFVo0ZG2Vm3g$Qyw4{A%0?WRnsjU(G?eU2q'
    'a{+v}6k$l+>VV+`MutpU9CoYIbk>^_p%=gA^lE7yqIXlJS}@_}suE|Hs)JT)^>6P7~pz6@s^VX^%OWCr5E7W8~5Urr_-'
    '`F3#6)ZSEGqJ5()&<iChEg;6&Dl*fN^+d8wv8)kMHql|r?ZF@MRkF0;x7aUjn`}$9?kxDl_3>5`wg(q@x5Zwy9}bGpHA~p~ucJD9nYAR'
    'hGJEay{0AHhGA1U*uu~FF<Ws^Pevt28(l=ju=jYwqxMQ(30m}Ej_Cm|VU}tU^2l;<m0(>3MgW#I{ae;5_8ryP{!d|k+I350JNqg8@fY('
    '2hTW#)O*SvK-'
    'H$0nwD!?{>fBQ3M6?;68AdD7{3C~RqzP)9(cT?b({USR|u1h+(i`uuWg@jeS7J4i%g1d;<#p=oo^>NCd$YOdLJ{7BIt^=;+h2SCvmmhF'
    'Wpufp${0dfZ{!{^eIvrH&soUWX;at5$=)KZa8V6JMis)g?MqFSI=o`o>I^}E?=tH!SPavhhQob^30c%xlOgVI}xElXYtw@pN48F*5Nb='
    '@%!>SUq%(c}eMmf(+MU`(xaR+Ihp<PsR$wlZSt(Y1jQ(7q^tk1<hv#ZEc#1&<@ScLCDOOcXPUi=}SrP}~;@;TcZtoz=HaBl=sD;Z~mbn'
    '2(}hOWisM9TP0u;CepECW;{5%0_O4VB|8>K4JpzmT)B1{xbS8IuxT$h-'
    '2KXaFf@cEP>)y^U@HLyd0C+}zr#<Si2SgnA)8;d)p{V|wU5k*Ozz#`>$uH}O(H-EcNuZ2uRn<e-7E$}w#K^UQv7ElV&csl8Qq3UA5!E-'
    'T(d&Sk3*1DWcWiJ27auYO}51C-'
    ';{@BNZUe3ZV0&QY}hiUE|j=uO${iom#YlDH6g#N0}$3mjm(@VC62!(sFYtYikOkELt+8_!hyRA?)GD^w=(nt1|M2sD7Tkf?tWGRe5TYi'
    'v}&4wi!3;4_>}k?r6VV{RbCtR|Y~!;Yka4r0K_{->M0uI#WpaMlPlRFR-'
    'F@P8GY^?MWt*M<wh3GO6;$YynBcYG%T!QG`niX=FJ2A2}tDHh~K<Lf)_D?x(0OCbc;0);|~!^ih0oO3<rx$gUy(}?(sek?b$j3SPhs|F'
    '6ZyC!d@4+oN&KlydieFT!f&_T^-dx0l2zYtm4f6P+3A3m<eK%}u8DV^q9@hiAX>T<%zze)N-IA!d?-'
    'xkw^mC|mSU<{rb*hcSA?v42<Qq14sdP{hCqj3e;%v8+H<p&8ap9V%1O4dVoUsl!LQg<+zas$80JDK{T7bs*6SdeRk^!Crj!JmmbiqvAV'
    '@y<kRLpNThJE!~2reRZBp0H+F7FhO~Z~~!E_?oi!s6Baw_B4c(hS9YGqx@d1jk!pdVcP=q#{Xeo$ZycjL{)-'
    '?ZvbC~x{00Db=&}AH};Qsl?zchfdrGvUczF9M_4AjRNAep2lO^ZV4Jb8LUmybmuiZ({mFFo^^|>Jfp;%Z6+0;%F_lA2`B=jlay{-'
    '#vS^V2f~6<A?B~Te?6EXT7-d+1tqe{PSv#Rd*!{L+Y8q964TQJ4Xx|FkBh%UfwQ!BmYi=Ta{z(LivklshY!7V~#z<$}2KzJJM(IS-'
    'H4_f4)J4FZkb1~MXd!Y9KFe;yo7+F2qxcWtpKuO>2C8z2!3^n$k`L#_)&@1wP2Q#6>7V&iT?JSc^t;v>N9mVnwlYbw;H`Ym^s7yvbPjq'
    'GXbOJwkLGHDk>YoJ4}FE-'
    'v#(!rBDn<l?%xJ)Kx^qH6XT#7Ay}+W7)6N8QUmNMGfiG6k2$7+gsk(=fg_;|dJTIp(AT%zQ=l9rUnW(LpAvXY7ZBIk5qNjoWtRoHCM-'
    '#+O|E4&(Ldm@{k$F{hXXT+dbV*^MDIYRpfkhIt@e~3XrdHjJPS=V_cr#h?=L=K>Oohf)4}U(Gbs&O0y4-'
    '|{~!J=;S_?3qU}F4lW11j%6c7Wr$xYTsn_HcVlmRo_z^iO1I7WqM0lUnK>tN*&Gch4jBHVwGSoU*TBXEMGx2Zu)S|82d%&L`L3M?9gAx'
    '4PLIdd(1mo7g5Td?(A@mNJi>>gFvNsdmL>D+71LVutLZl-'
    ';Lhda8z)}LR^@qG0naXZU+#>J9Dw20GnewdC63z#Vw9Ewh;5CSo^l5JkxXjyMYQsF>{mKfhPGBov1^mR61E6th5JUEQ<4gBCI)qVtl+;'
    'Gbl~)*V25te*v5`QU@)UlHf7iVge(~1TJc)lx@pe4XrF4NVLve{Rkbq9}b}_w>a=d3COVHtjfzH@w{X$zqISU+tp2H5iJ_dXG*9W4o&p'
    '>843~b_F3u`a{=w`oSXiDxvM{DiOHv&z}+oVbG?_gW*T<QqAd8iZm9}<I9jccI=EzN9?l~#%WgEks(@sp7%;d+64wg~wm@C71mL)c~bY'
    '#jBy76#*w;v0p+{2*m*ps)6K_*eYBZZ9WbUckoA32YI<##rzK+!fD-SG!I?9@`YIlQIU}A-'
    'dsq)G={Scp_Dwo2I(xX31muQmu`(6r?Fz@F7H4V9W_8<CA?=J!<Hq{9j-=JynjU5WEq`!nxv1`*-'
    'O+eV&_u>Ajzk478=T6w;*#TqEeCe4g*cU6(59!QeA0T66kEd8#ztYZ4gXDV_`-'
    'F@y45n7~>J<H3#gdlnP(o8dgaj2U7#;C=C)$ZGJCOfg&OEqofjn9bxN{*`o7ZcLU#FVN#E?O#G&aM-jds^a|>eU`iptcguj>O23Z%|$Z'
    'shDbkKjy+#ZA-b?L;sVNl)G%cry9l3yT|q85KEuh_RF1dIa+KMACdquP*kVdzr*b2UHgo+_KX`-'
    'UXs!o0$UMM*j+@Tb3Eo!jnd?GDu5D;JGFj>rXQ#^(qI@Y}UAS6dery@Pz|&s{m?D8ua2~SXErq7QpQRl7Hci0C#Wujxl72uw^3Bu-yA_'
    'uZ8^L8dRezrTjBgD*0Z23NfnB;^kpa-'
    '0{f)4?5NIq%?beCLd)8KxPKr=0*rxrGzZ%xT(~UcWiO4BmS12AS2u`xKlIyZztQJ0^DAIZY&qiX3tHL9Sc5t&&M+W~Try|GUK3JCh6FL'
    'ifhi90Efwa4UJ_gI!H{3oD?x9>mxx!WQIQXXgi5srG4rKB{Wl#bvZIea{Ppq%(Y0~OI3Gm$V0xiq`2gt`lrs4e9w%O<-=(GI6IFg);J-'
    '5p6;<)otHQOu7EdI&PcKz-86uL+~fKN*4L<w=r>B0`$C)zvG_TXs0OYMSZa#w<vyi-'
    't&97A0bU&$mq%rz@{WFQ6Fqh2j`qE~r<h`_H)@!SgJq<b>t1y@Q1(rC<{^i`%Dp7__EzbsX7D>6fTBJGDt?4wnO(HmONztq>Izu~=tKg'
    '{V=Tjc|`lh_aCXgS!glpD|I^X(bRC386s3z5(uqP5yde2;bVo=UooC-ZNd1ZI*NP-'
    ')U>BEfYRc3_uGU*u?Q6I4NzK|@{duuF;vw<m|_yCfaMYVlg=D&bQyy{*hcpytqL-z{v2{D!y-'
    'PqZV<BC4Blw)Khce!C!=7S40UF=hU0(t0LJ``0N+K`;pkQV*0$a02y>nkx?z&%*KC3}gfRNU9Crwhe{%!u8!>h2P<qrcT1ulrXT3-'
    '0Qx`cSd<E0jeqYL%sZFD3d!2r8CPML!h&=EKRdbGc`b`6BFZ}nKQYX)<9xQp^xsGWg4LKcUQ9bj&alRJnJ}Hp~q|u`^QKyT7uifN5Rd`'
    'Z=q|f9=OAHR@yT=QaSdjRv()|7JG*nHZ{K&euMODzc@HcIfPvn&giYk8e=wlN^cHBbiBR-|1EzUB49(f)V;K)o1(YW^+X|QV1Ib3d@7-'
    'pb*=CZG{Ble4zO*Ja{RZwQIuZmBd~lest56>uvlxE_yu|0rqxf;NTy(G$2N=k{&06X2YhI<_#ALYAho2AzyV7!iOx572K7>`DMvX5o`L'
    'I2e+g%(c8bl8`*6zP@1W~M_3{1+ZTBR2UorKp%Zk5HJ^cMCKQ+en&T`h+)V$U`-'
    '9Od+FH=f?^4T2i%{+K*_kCaE_+9#h_(tB1o}F|X|3JFTvy=Le*okgt(X++o{`4kl6Wg8MK=p|FVV-'
    'UBC*~Me`Zm$B_a)ma@i*!ebtCRS&;0}obq-kP+g~^ZG<na^hw09iTzZNpUN6y`!MepOO8(jRs^IU~r_?gjME7A+D^m^c9bn0x^|1^5&8'
    'U#+nZ88zN`LU@J>-G<=pA=tQz$gSv&8&L9%zpw!qSk?F02Ue4P4aEb0Ab4bTPHe+Y3BGM=*1tW|)VVMc=j5w|~WS;%B<bp-'
    ';g}_8a2&q;o_Y;{k3qdl}MWl+plQ!>(wPOOyah(GT2c<TTv_9m5U~Cn{O$M|B4tL-4lxa$nCKZLikC(gf_oP1ZfPJ;VN>t4OMpU@#I-'
    'b<L8edL94``JUQSVLx?|7%%zlABYX)+c-ng4R|i!&-5(R4A&tixXI!O*6W?mw-*PBvr<p^c9L=S$A&5}2aMz!+q-~WtiQQc!$!j-'
    'CXHDReo&hl%IHQ&CY3{tlbQwst^uaTcw;QXGMc%CwJ`4n69ci_D7mR#uk|;7rry%);dgL?w3@49k0qyACHps93@SlL>}2{9Kb`(g_4I$'
    'l^MFWe9k8!M^3IjkiSs?j=^D~1Fbhg!sw)QPN$wM}6?)I4^DW>la82J+jZVFx#^QgG_mjIyN0SByT3C+<Z;%8sDZJ3vB>4@h3smuZa^='
    'Zc+-'
    'q`EpqFQrxGaGsMkGzs2Cxs5y3kI1Y3PYusTL51q#hEOlp$ogZx($QFQZq;7bIj&ReCxwA{j~|#wvyu$)OZvzTpKtftqOk#x;c2=!;l_-'
    'bHO82UvrazWheAx<k)3)V9!z(B9HyTP^!(DUCRt+>q%HE(;X%gYbvKyI=%)HT6jnfz?Yo!#}sVk?nGYvs>x{=MN%Vi-'
    '2P2cSNeSYhb(Gkd~k9qOQq*(Gwh*S|V9syPRvYX6R0+bHN3P7*wV_{kiYY!VB+a+h_VRbO@@FoJR`4DXDdveVRXMo8Xlr%uY`^aTMa&x'
    'ekcS<r#S;P>#=Jmnk{`2)B?<gsO_W+`lko7{;8!#_qEKiyDnoGfg#^Y0^tJN?pb!aUZ0*%yO=dl`WXU-'
    'z7(Kr}55~4!*yktw<z1L5s##V6SoU0OA{+wA85Z>#&jB9z0GvX0Bt+7wZV8@B#5te|GYFupRqT;FPM?DttDT&Bg=ksO!Wi%_e@thDsU4'
    'v*e@V1gLp%n13MW&{{I_pj)bfU6l`be<APj)yxgSRr2lFI1Pm$Vp(vGoFztw!~{khr>pSAlL_cG{UUXn){T0_Rk!}@jo|0Vdy>-'
    'N;X+oD8RflQw4kmA`Gfk8e*CA#ZBgq|-'
    'Jx6^11HN%t?fcfp;E!x41yZdJK1#FD<_fq4tp6{{(+!Wp|}`rX>mz$DXn+U2YXUO60(^3W~aTDy{Bv@cZoRr9BM7s4%gE%v})iN><k!G'
    't05JdS8SNP44W&k{z|kvFw_oM?=p>?QP5*!Q+^NrYp`ah9LiG*#Tig4_12ah{Xcg#;xwXB&){89dvoIwf?g*sRUxIovjMrHwCB57j~R8'
    '^I_{vgF4dce3?-mP>@lbn-W&Kun@IOp+8AJD7JVtykL#<o5g(^+(GWRXYK;BkpQ%MMVa*l1R`}f80-'
    'R!+5Ofk>Q=ZGig1EPiMwXVKUX9W&l-'
    '5&=ZIhA?xkqU0a1O7nL>tT5Md2ov6fua*2fKkK@@YfI@C$qwx{GYUz17;G<FvVu5!?tik?+JZU{z#6X`@|lyz|wd@e6Db4hD}T2IZE8o'
    '02DB(})v4`TQQVOQ=Adzyj4RUBdgp8-'
    '7}1KPcM%1S648;CK55zigi@HRq04v*aSRSyB!4rZUV0BNv5kmQHY3nCZ&1yw#mWIzYRbAkdOqAhou>2ZlO6$YuOPxL5PeL=QRB-'
    'O5(Ri>4qp06A%|32k!Tg`UEV$@9`|$h8}mZrT>&US*K9Oy7l9kWJEhN$)6@9z(n7m&QsV*H;}@q%W?H$UlZRq`>;P%eG2vovDob-'
    '3P#(l6T7+*(yXShGTUKT@=-JySs<8u?;o#v`&vrc6Lepi;dyjrj6{c`89kO=yyyTF4=vNy-'
    '00m+E762?A9ARqd)_F#M8U5r?G3veBUqT?aX@4PN&l+x$W%f#EO!gyVE@)UEnkL<1JI7rceN2EL_gm65Dth>zl9-'
    '*vW}m?gQp|j{VNYjNR8*--u}iJ}D}0ei!V;)^#qh9HSHXQp+sL=f1tOw!SJG?VAELrEVG4djIxka@SqW6IxJFT(7-jO_NnWF-Phv$e5R'
    '~(4X*oT&B0O%@AaEuqRTKlqi2D77L>!5D~a~{92D-9WUP~y2ju3-'
    'jYt(w|XKVRp5~{e!2J#^GjQ#22fMwxox$U<*#(N!yXBDY)0}7@mW?4U9j%t3oyaim-@Y!mTTJo!YT+%>jZl3Vc-'
    '&V!`jjD5&0VzK**Ra6*`)rcj7()k@!pJ06QucOP7(V#u-'
    'Ub{1Gt|YbQ0*9z%M=w&*ePE^<KeYpRI4tIzhYiWvl`%s1+c=K)<#uZW+|W_cel?znS#&kQ4tmcr%KCsS;k-tn07Fc-'
    'X=sE7IzD&K>dXL=6kJNRzuCnRKh4kVuP9&|*}x2TqerZmQdsoez~i@5v?=7{*}%#1=Ro}%x09=a0R&N4-U?JVs}-kWQ;tV8wFmE~8}-'
    'D1jscCNL}deKw83%q$?n7-=uP~DhiyDw0(UWo%KoZ_N`z9N59#-'
    'M9Q_fG6&aubzk_0Zi=acBlIF=@B|yS$Glv|38N;CZQfyVhEc=~kc*@tZV}JR1`Xlq<uuF5a`YQ*@`m1u`q8_Msc@S%$CFT5UAEzcjD(X'
    '=otd+&VWo!%!>O&ls%@GPN(ggI1Q#kFTq5gkFS>g4>N&@OZe~`ZuAVFT&Y@eA<d-TjRW0yF>gsIg0+rYE7-ycC6>RBmwE%V&QG*8P-'
    '>5N_<ru*kEhsfgE!*w}o$x)+JQVP+De>X}cfFM=lsjp$z1())2D-'
    '^^`&O9dLVOFw{8JA&(KRNL!WqWQ9CgT7o*1t%22aTkZ;N0;XV%0s~WTICg@)xjqsDFVw!ve_|&Bhp>zwV>~3ZVBd0ikmj6lU<ak@Qc0i'
    'v1JsC`D$U`}Bprt?TON?b%p!QCeWH9;9A}(NT_?9DAJo^e4GmdiryIle4C)m&F5Cd^9Q9S{hrg7{?K)8^-'
    'i<8q9tz|LEAXd=#(@oF@PN@WRlB22^4XoySPTD}@J7iHJ%bbQ54r>N7k!?11e**0PnEn@$fsn%e_<L(%1xqM8TUY~+1s*TItI2de`QYk'
    '?*e~V=kc$6FUc27lD`%F_$L8nWAjBFoSYxn`hdwpesQI;4HXMt#kR>+R~i}mj18whNWW+}T!%5yJFs`+VcmCRAwJ#rn&}<r;Ce2%M9yf'
    '{u}{Db`XD-rs*bF8x02suGeS!XXOa`(qs~T97j(5TMQRcz@fLF1)Gv-'
    '}WJmIE|7oE`Kp+!Rk2Pp~=sP*77}WIEv&P<#lcI$sz9QQ<pslwUhxy1r3~&JH1jnfkzr^Ni!-'
    'zkj9!UwdyXI)U7sTT1#;T#JPzU$|fDjX;PMDPou{~m@vR~nz$_A{M2*NXkrr2CF?K@*lPiUt8MzyAwCG9iiLdT*Vg&(2ac%>2rZxJJf?'
    'qXkP0GGOd6r!s8ktZOM9830q3}7JdJ@u*H5yK8gUF~M+Rqz@V<zI=y{^!I8`vUti4QaPECXt?@#FQ@ezCfy=cchMVJW^cjQo3=w2py5C'
    '+r@tY#~E(Y)dLmKY-*7-'
    'lo&}aXgma)0m0f}yuYLi2H{}bA!ks?CelsaZT)R0oN<X6c2wKpaUgwz1BsqkOHT!LBiLI)HuPogtKSSYfGldD_Xs&hX#=j5*R$EyVd_T'
    'iocSoV3_GmOBGdIoj0_ZzYe6Hrl^X!*=xFIKRvgMO9r3y)qvtu)0~gRtC7{M@2`FQ3jFj<*k?}+fG?tDs*AgGPSgcl3id;q#%sXZW--'
    '?US<r<sImBd@&u_H6MXfFXThyVY3XQ7lVrF&;uX5a(3YNAEXlE-1IkWTn*tdH7L=}eUJ+p%YAq_4vGnT)khG8?%y+DF%B@P~B57-'
    'Q0@2hn({y|Plv=idk)rI^@^#If>vC=GzbRP`dU5LLyM*bFIO^7A}8O8pLukTRid+!@OSE=8RpW=e79IYAH@Zsm;vXOSZOd)bvm71&eDX'
    'zwKtBmWBiVBOL`ay9Q%n?qhAC*cPi^AwLpIJSG+!q26qN`LG~$cwiGbD*tkCAGxYfLRBP+gFL#M|MzYmZ4B4(+o3U0<_*28@<`z&pV5)'
    't>n<ZLiIg!eJr$xyMxtZzPP8W)0HLYWlILv*|A)ULMpJ3oy9%^FX4rDyHpe6fXhXr$S_nTx)Zb+oB^59W55mQg7&%aG?9X=<l~VC@GIj'
    'cWV3xFJe3;dyhpEuFQSpkd&F%WW!Y*Q&+TX9nU<bKM0dD3xuJ9uSO(Mxm^|}+-'
    'QcU#4_jgUZoDN%VKXI(&<Q8uzW#r)VZ?eW1I|U?p}7fVQVtkT%p=z5AIN<qlBzUSm=pAnZmGX1*pu&ITf#O=n$6x+j{~5gHntf)|C6R_'
    '07syP@)1Xo;Xd`rKbPL+uVU;>A9h~w_M~bu4&6cO9knr*wN?PfcIA13)O7ca;sWNCk25umTnCP#2Dn?%brbBgo>80|JTK`&-8A-'
    '&PB0Z4_xYCsVMa17pl^F_>Q-Ayed$2YA}iCC3RxbT3Jiso##{xH<-'
    's|O8tkcAP?=CdH31Jg%Z<B%2)d@Nr@04RqxiHJGrFmHKExBtEe1xoI#Gz%nKzO?Y4A~*tVRv<yTMPUa`QT-13j*!wx^T-'
    'Ha(cF>c2rXk1xo-YZ`C9VHr<<F|IR=pwhgDjrs05)b70h`MX&Hy6Ki%o=CcmIm<J3zmw*R{f;cUkUkoJ&-BW;I^lK6aFg4A-'
    'Bbp2WoiRG3f3^2JpY<^`1g7m`b}KP4s$}WwGO+=l4ABd4}p49b*5dkKXHfqiy=DZ7L{RsXg2JvMpaTDn}2e}_<y0FTShrsF`eE2GJahg'
    'b&l#vJ&RAIv$+?ZbLNgEAP@^R$vY(&z}LW3`Ysp{Bk9Kl^@Yt;1w9aaT0BHchlg^PnRMT6AWNT$MX@`%0tQn03cuLAVs&|hXRjqoPLAE'
    '@U-L7!0GLAGjx7fJ3KuyRo(EpGE#`CNG{MIu1m;sCk+=DcqW@rimTRk9oMyU=q|%<B)S1FQgL0Xc(pm64-'
    '5e;Rw+nyTTw)hbPxDssu#|;`edS^$b5h!%eDfA?FE|@DVFw4bqJPjiL=@2pPgjfVb?lLWjm%N}Y|>SAoB3o)p;2~c!a3$Tfp^BJFahf9'
    'QGrWx6=bM=4DkY+aUd9c3_nP{9+uTf_Uea5l+G>PtL54n6Wam{@FJr%{3O_$m<K1@`x8g_S|nqN4kRdNNQJ~T6Fp7qjdfMGxSal+a6{s'
    'iTu<77h1K$4B4QM#pqolm&9BTSG$cNh9rPYJ3t_;wz<Is*aKrFLv19PxKvC#|JOtV9T@QZ7bD>JJK`X?v5WjG!^a3^}sUnC|)9oo<$`^'
    'x9vAj}$GnWU~6qCd%rJnp1nxpFMHHrH|zhD9qlaw74h3V$b$_I#MZm2D-'
    '4T$~R2`mm2`N3KN(GKqQ^J+(GK|EdGQLW{x3Js>~YfXHy!E(G?=}yUuJSqK-'
    'Z}hgO7Q)5yh+rzDm%BnWy%&;8xS{q8d!GYeC}Vg6e7$f}sGng%czNkDvZeOi)LKp}x`s@kX8IEG*5ColT=_BDQEJP5DK$fS^hF>MxItI'
    '*_^C3a5^#q48y^Mdxi;Xte5ec~ZoW2lAAJ(qsnT$xr2hprvquwNX-n|Ij!V8;=r-'
    'g@;cI+xVg?9k8Q4>(wQT^t@i1m@?DGY5Y%i&ev|n0k+A4j9X9vvE83YcLqif_#N^Nir{V)K)DdJ5SG)0g{#U#S1e<*ectk;U*7KTv)Ut'
    '$&Jj{w6>AWlmUlsoJk><mzQ{}G%6M6jhG5RH^%|H|O;_$!>j*Tm@8a@}A_4KzLGnz9XP6nx{O{Jku*cpLOdio^f5lIk_NOI|O3<xfVO2'
    'UV(lk#cmT_YORlPq2@{JQh{DChjm*m<D;gw($q6Lv`FzfW&*sU#OGpF%K+`LbnDUvcs{#>{8oMHD0-'
    'CX%sidyaFvRR&2i3Eo@IH8gg1ot#8Pqat35@uU02mx`+A&wz9WDzxShQn3Runrv50Ljs3w7V6K&fO?QAwI4sUL3+!UR>_4s@aBXOg=s9'
    '2A(7(t{_DVtw54%Or4d{(J2plU75~uPnkix(Z&B}}|@~{vD0?p7(wi?<*X1;YMxJ23ps1nDn0;-'
    'pM(sb;4ApX!q?1Pk&xXHKIv_)to)Hf_=4#3-'
    'xEA$!%8xsQ$Y8j>I!L{gTI7PPxd79W>lz#TDVtQ1t30m$LAVDRu!ROKndY0HosckyME{D&$?iSxkVC9nv2HKT{<jbF4_>a_(uYw(lsu3'
    'J-Y2o}PEH`}#-'
    '3J;_QxgAG$72oQ9<~7tMc0*P(^sHMYmT%?Sq?^UtFZULGvuPF!f{;vE;JLTVFq!q&|17A9$@5xbKWJg(d!N$<5bU3;4;u`{~TvEG1k|u'
    'MA943i^6(km!mz9Pq*W!m=v@xA5EFKO3<$!u~N2G+HE8PsFvV2t@TMFAiRg$o4Vswu`<A6jx+u!xy0XqXG<YqxEO_PMYHkK(yX8!oE-l'
    'Kjdn~k2jpIkeex<}FKh(0N9_Q$gWTeHdZl(mKE>Oyp<GAEO>TyM<9hMD#VS-m2T-'
    'h3Tf7ql^rzs+#JSoe_7wF}9gFn~f1+*b6I(|Yhp&L+bvk$rm5B_q-Djepo!o9T-'
    '9A!W2`qxPFm?GA;QU6Tf{CzQ&XY{gceN|s*|Vcyi><xutN9u9M9wYpm_SRpHJ4iKtWVbo`lK~bAGmdJG~NW(;mcfqE9sV*aaiC1U5(nT'
    '9W_6KX7KHidSD-QyUmYFQi;8}I!Iu#G+PxJ4h<!8ExFP)s*rgNH#F_#48$iaTsmKBi^5ntVYO5v@J`kd4b|<`31YwU$pfV%z=s`QF)|!'
    'WrX!7&x7MNJZM4XHoUgFuqH+i+{F7Q|uqz$07UA^3Ug!u}8#RgRxpc%zMWOHTS(w*Yh+5EQNG6}2@Vjwb=~pa{Tn2Xz@%D43>1b}AORf'
    'XYw^Ra-J<av*A_volEDGi#O=9Y5pJP8-w}s~ruY!7fFfjtpQ`ZZlh;-'
    'DZo+d!<qEHUCls6>(1%*q0cm0J<!mF`Eh+(QBtYX^^P{xNT*@@erb@Vi7DN_fTDt{)XU=hLgP*rU^>hgq@F=*yb)|-'
    'e7l?c<WaE8E@UZmF>Pjm73F9Dl3L_`AfQwQ5wb&S2QTwsmF251d|hCZmYx$09c;!R4MVD*q(`We=sw8Yjv@1r%qNZzsjkMYCFJ?e0rFn'
    'odEU}kJsIM)8Cv<caQ?4MLC+>sfa`VCtvZ?u<-8vYO-sdY)(Y}^!Vr%~1z@({CE7$QA6SQ~nyMfnQte-'
    'aVE7<mwiM9;@YaF5(3bhkFw*F-2I=8>($A#eli{(%IjL1@bUfysS;&N9%x-PAkf-xO6dI-'
    'BENz#8Oe;-(+-uP5KJTiuHSqhk#8JigMp5P1i60;btc7)W;(yF&Uuz>bK>'
)


def _read_weights():
    raw = zlib.decompress(base64.b85decode(''.join(_WEIGHTS_B85)))
    n = len(raw) // 2
    flat = struct.unpack('<' + str(n) + 'e', raw)
    offsets = 0
    layers = []
    for inputs, outputs in ((23, 96), (96, 96), (96, 10)):
        weights = []
        for _ in range(outputs):
            weights.append(flat[offsets:offsets + inputs])
            offsets += inputs
        biases = flat[offsets: offsets + outputs]
        offsets += outputs
        layers.append((weights, biases))
    if offsets != n:
        raise RuntimeError('Embedded neural weights are not valid')
    return layers


_LAYERS = _read_weights()
_LOG_81 = math.log(81.0)


@lru_cache(maxsize=14000)
def _score_after_round(dice, starter, you):
    """E[final placement] from start of NEXT round (relative player order)."""
    n = len(dice)
    if dice[you] <= 0:
        return float(sum(d <= 0 for d in dice))
    alive = sum(d > 0 for d in dice)
    if alive == 1:
        return float(n)
    x = [0.0] * 23
    for i in range(n):
        x[i] = math.log1p(dice[(you + i) % n]) / _LOG_81
    x[10 + ((starter - you) % n)] = 1.0
    x[20] = n / 10.0
    x[21] = alive / 10.0
    x[22] = sum(dice) / 80.0
    for layer_no, (weights, biases) in enumerate(_LAYERS):
        if layer_no < 2:
            x = [max(0.0, sum(map(mul, row, x)) + bias)
                 for row, bias in zip(weights, biases)]
        else:
            x = [sum(map(mul, row, x)) + bias
                 for row, bias in zip(weights, biases)]
    eliminated = n - alive
    valid = x[eliminated:n]
    peak = max(valid)
    probs = [math.exp(logit - peak) for logit in valid]
    total = sum(probs)
    return sum((eliminated + j + 1) * value for j, value in enumerate(probs)) / total


class _Binomial:
    __slots__ = ('n', 'cdf', 'moment', 'pmf')

    def __init__(self, n, p):
        self.n = n
        self.pmf = tuple(math.comb(n, k) * p**k * (1-p)**(n-k)
                         for k in range(n + 1))
        cumulative = [0.0]
        moment = [0.0]
        for k, probability in enumerate(self.pmf):
            cumulative.append(cumulative[-1] + probability)
            moment.append(moment[-1] + k * probability)
        self.cdf = tuple(cumulative)
        self.moment = tuple(moment)

    def less(self, threshold):
        return self.cdf[max(0, min(self.n + 1, threshold))]

    def expected_shortfall(self, threshold, cap):
        if threshold <= 0 or cap <= 0:
            return 0.0
        a = threshold - cap
        ca, cb = self.less(a), self.less(threshold)
        ma = self.moment[max(0, min(self.n + 1, a))]
        mb = self.moment[max(0, min(self.n + 1, threshold))]
        return cap * ca + threshold * (cb - ca) - (mb - ma)


@lru_cache(maxsize=700)
def _binom(n, face):
    return _Binomial(n, (1.0 / 6.0) if face == 1 else (1.0 / 3.0))


@lru_cache(maxsize=1000)
def _binom_p(n, p_rounded):
    return _Binomial(n, p_rounded)


def _matches(die, face):
    return die == face or (face != 1 and die == 1)


def _shown_hits(shown, face):
    return sum(_matches(d, face) for hand in shown for d in hand)


def _known_hits(s, face):
    return _shown_hits(s['shown'], face) + sum(_matches(d, face) for d in s['your_dice'])


def _next_alive(totals, actor):
    for j in range(1, len(totals) + 1):
        nxt = (actor + j) % len(totals)
        if totals[nxt] > 0:
            return nxt
    return actor


def _minimum_raise(count, face, new_face):
    if new_face == 1:
        return count + 1 if face == 1 else count // 2 + 1
    if face == 1:
        return 2 * count
    return count if new_face > face else count + 1


def _legal_bids(bid, total):
    if bid is None:
        return [(n, f) for n in range(1, total + 1) for f in range(1, 7)]
    return [(n, f) for f in range(1, 7)
            for n in range(_minimum_raise(bid[0], bid[1], f), total + 1)]


def _preferred_faces(s):
    """Latest face bid by a player in the current round, or 0."""
    indices = {name: i for i, name in enumerate(s['names'])}
    preferred = [0] * len(s['names'])
    for name, bid in s.get('rbids', []):
        if name in indices and len(bid) == 2:
            preferred[indices[name]] = bid[1]
    if s.get('bid') is not None and s.get('bidder') is not None:
        preferred[s['bidder']] = s['bid'][1]
    return preferred


def _face_prob(player_favored, queried_face):
    """Slightly biased per-die match probability conditional on an earlier bid."""
    if not player_favored or HISTORY_WEIGHT <= 0:
        return 1 / 6 if queried_face == 1 else 1 / 3
    weights = [1.0 + HISTORY_WEIGHT if _matches(f, player_favored) else 1.0
               for f in range(1, 7)]
    total = sum(weights)
    return sum(weights[d-1] for d in range(1, 7)
               if _matches(d, queried_face)) / total


def _unknown_distribution(s, face):
    """Poisson-binomial distribution of hidden dice across rivals."""
    favored = _preferred_faces(s)
    distribution = [1.0]
    for i, n in enumerate(s['hidden_counts']):
        if i == s['you'] or n == 0:
            continue
        p = round(_face_prob(favored[i], face), 8)
        block = _binom_p(n, p).pmf
        next_dist = [0.0] * (len(distribution) + n)
        for k, pk in enumerate(distribution):
            for j, pj in enumerate(block):
                next_dist[k + j] += pk * pj
        distribution = next_dist
    return distribution


def _call_value_exact(s):
    """Evaluate immediate CALL with neural V over all actual dice totals."""
    if not s['can_call'] or s['bid'] is None:
        return -math.inf
    bid_count, face = s['bid']
    bidder = s['bidder']
    you = s['you']
    original = [n + len(hand) for n, hand in zip(s['hidden_counts'], s['shown'])]
    known = _known_hits(s, face)
    dist = _unknown_distribution(s, face)
    by_score = {}
    value = 0.0
    for hits, p in enumerate(dist):
        if p < 1e-16:
            continue
        delta = bid_count - known - hits
        if delta > 0:
            lost = min(delta, original[bidder])
            resulting = original.copy()
            resulting[bidder] -= lost
            loser = bidder
        else:
            lost = min(2, original[you])
            resulting = original.copy()
            resulting[you] -= lost
            resulting[bidder] += lost
            loser = you
        starter = loser if resulting[loser] else _next_alive(resulting, loser)
        key = (tuple(resulting), starter)
        if key not in by_score:
            by_score[key] = _score_after_round(key[0], starter, you)
        value += p * by_score[key]
    return value


class _QuickModel:
    """Fast *relative* EV ranker; rollout+neural picks the actual move."""
    __slots__ = ('public', 'my_total', 'next_total', 'next_hidden', 'rivals',
                 'my_hidden', 'other', 'opponent', 'next_dist')

    def __init__(self, s, public=None, revealed=0):
        you = s['you']
        totals = [n + len(d) for n, d in zip(s['hidden_counts'], s['shown'])]
        next_id = _next_alive(totals, you)
        self.public = public or [_shown_hits(s['shown'], f) for f in range(1, 7)]
        self.my_total = totals[you]
        self.next_total = totals[next_id]
        self.next_hidden = s['hidden_counts'][next_id]
        self.rivals = max(1, sum(t > 0 for t in totals) - 1)
        self.my_hidden = s['hidden_counts'][you] - revealed
        other_hidden = sum(s['hidden_counts']) - s['hidden_counts'][you] - self.next_hidden
        self.other = {f: _binom(other_hidden, f) for f in (1, 2)}
        self.opponent = {f: _binom(other_hidden + self.my_hidden, f) for f in (1, 2)}
        self.next_dist = {f: _binom(self.next_hidden, f).pmf for f in (1, 2)}

    def score(self, count, face, own_hidden_matches):
        kind = 1 if face == 1 else 2
        known = self.public[face-1] + own_hidden_matches
        others = self.other[kind]
        from_next = self.opponent[kind]
        gain = min(2, self.next_total) * (1 + 1 / self.rivals)
        penalty = min(2, self.next_total) * (1 + 1 / self.rivals)
        score = 0.0
        for k, prob_hand in enumerate(self.next_dist[kind]):
            needed_other = count - known - k
            bid_value = gain * (1 - others.less(needed_other)) \
                        - others.expected_shortfall(needed_other, self.my_total)
            need_from_next = count - self.public[face-1] - k
            next_call_ev = from_next.expected_shortfall(need_from_next, self.my_total) / self.rivals \
                           - penalty * (1 - from_next.less(need_from_next))
            bounded = max(-30.0, min(30.0, FAST_CALL_SHARPNESS * next_call_ev))
            prob_call = 1 / (1 + math.exp(-bounded))
            score += prob_hand * prob_call * bid_value
        return score


@lru_cache(maxsize=24)
def _reroll_distributions(r):
    """All different 6-face count vectors and their multinomial probabilities."""
    factorials = [math.factorial(i) for i in range(r + 1)]
    numerator = factorials[r] / 6**r
    combinations = []

    def rec(remaining, face, counts, denominator):
        if face == 6:
            hist = tuple(counts + [remaining])
            prob = numerator / (denominator * factorials[remaining])
            combinations.append((hist, prob))
            return
        for amount in range(remaining + 1):
            rec(remaining - amount, face + 1,
                counts + [amount], denominator * factorials[amount])

    rec(r, 1, [], 1)
    return tuple(combinations)


def _screen_bids(s, quick, legal):
    own_hits = [sum(_matches(d, f) for d in s['your_dice']) for f in range(1, 7)]
    face_scores = {f: [] for f in range(1, 7)}
    for count, face in legal:
        value = quick.score(count, face, own_hits[face-1])
        face_scores[face].append((value, count, face))
    shortlisted = []
    for face in range(1, 7):
        selections = sorted(face_scores[face], reverse=True)
        shortlisted.extend(selections[:3])
        if face_scores[face]:
            shortlisted.append(face_scores[face][0])  # minimum bid for face
    shortlisted.sort(reverse=True)
    unique = []
    seen = set()
    for score, count, face in shortlisted:
        if (count, face) in seen:
            continue
        seen.add((count, face))
        unique.append((score, {'action':'bid', 'count':count, 'face':face}))
        if len(unique) >= MAX_BID_CANDIDATES:
            break
    return unique


def _screen_reveals(s, legal, start_time):
    hand = s['your_dice']
    if not hand or s['must_raise']:
        return []
    groups = [[i for i, die in enumerate(hand) if die == f] for f in range(1, 7)]
    groups_lens = [len(group) for group in groups]
    base_shown = [_shown_hits(s['shown'], f) for f in range(1, 7)]
    by_face = {f: [] for f in range(1, 7)}
    for count, face in legal:
        by_face[face].append(count)

    preliminary = []
    for amounts in product(*(range(length+1) for length in groups_lens)):
        revealed = sum(amounts)
        if revealed == 0:
            continue
        rerolls = len(hand) - revealed
        public = [base_shown[f-1] + amounts[f-1] + (amounts[0] if f != 1 else 0)
                  for f in range(1, 7)]
        quick = _QuickModel(s, public, revealed)
        # Risk-adjusted mean-hit screening; exact multinomial calculation later.
        fast_score = -math.inf
        for f in range(1, 7):
            if by_face[f]:
                expected_hits = round(rerolls * (1/6 if f == 1 else 1/3))
                for count in by_face[f][:2]:
                    fast_score = max(fast_score, quick.score(count, f, expected_hits))
        preliminary.append((fast_score, amounts, public, rerolls))
    if len(preliminary) > MAX_REVEAL_SCREEN:
        preliminary.sort(reverse=True)
        preliminary = preliminary[:MAX_REVEAL_SCREEN]

    evaluated = []
    for _, amounts, public, rerolls in preliminary:
        if time.monotonic() - start_time > TIME_LIMIT * 0.28:
            break
        quick = _QuickModel(s, public, sum(amounts))
        grids = [None] * 6
        for f in range(1, 7):
            # Minimum bids and up to 5 promising slightly higher bids.
            choices = by_face[f][:5]
            if not choices:
                continue
            grids[f-1] = [max(quick.score(count, f, k) for count in choices)
                          for k in range(rerolls + 1)]
        score = 0.0
        for counts, probability in _reroll_distributions(rerolls):
            ones = counts[0]
            best = max(grid[ones if f == 1 else ones + counts[f-1]]
                       for f, grid in enumerate(grids, 1) if grid is not None)
            score += probability * best
        indices = [i for group, amount in zip(groups, amounts)
                   for i in group[:amount]]
        evaluated.append((score, {'action':'reveal', 'indices': indices}))
    evaluated.sort(key=lambda pair: pair[0], reverse=True)
    return evaluated[:MAX_REVEAL_CANDIDATES]


class _InternalOpponent:
    """Mixed, imperfect policy comparable with training-population strategies."""
    __slots__ = ('style', 'threshold', 'noise', 'boldness', 'reveal_rate',
                 'call_rate', 'bias', 'risk')

    def __init__(self, rng, is_ours=False):
        self.style = 'math' if is_ours else rng.choice(
            ('cautious', 'aggressive', 'bluffer', 'balanced',
             'math', 'reveal', 'random'))
        self.threshold = 0.64
        self.noise = 0.06
        self.boldness = 0.0
        self.reveal_rate = 0.0
        self.call_rate = 0.17
        self.bias = 0.0
        self.risk = 0.0
        if self.style == 'cautious':
            self.threshold, self.noise = rng.uniform(.73, .91), rng.uniform(.01, .06)
        elif self.style == 'aggressive':
            self.threshold, self.boldness = rng.uniform(.28,.55), rng.uniform(.1,.4)
            self.noise = rng.uniform(.07,.17)
        elif self.style == 'bluffer':
            self.threshold, self.boldness = rng.uniform(.48,.75), rng.uniform(.22,.55)
            self.noise = rng.uniform(.1,.22)
        elif self.style == 'balanced':
            self.threshold, self.boldness = rng.uniform(.53,.73), rng.uniform(0,.13)
            self.noise = rng.uniform(.03,.1)
        elif self.style == 'reveal':
            self.threshold, self.noise, self.reveal_rate = .74,.04,.32
        elif self.style == 'random':
            self.call_rate = rng.uniform(.10,.35)
        else:
            self.risk, self.bias = rng.uniform(-.10,.10),rng.uniform(-.20,.20)


def _internal_move(hidden, shown, totals, turn, bid, bidder, must_raise,
                   policies, rng, my_index, skip_reveal=False):
    total = sum(totals)
    hand = hidden[turn]
    all_hidden = sum(map(len, hidden))
    unknown = all_hidden - len(hand)
    p = policies[turn]
    public = [_shown_hits(shown, f) for f in range(1,7)]
    hits = [sum(_matches(x,f) for x in hand) for f in range(1,7)]
    stats = [_binom(unknown, f) for f in range(1,7)]

    def false_prob(count,face):
        return stats[face-1].less(count - public[face-1] - hits[face-1])

    can_call = bid is not None and not must_raise
    best = None
    best_ev = -math.inf
    choices = []
    for f in range(1, 7):
        first = 1 if bid is None else _minimum_raise(bid[0], bid[1], f)
        for count in range(first, min(total, first + (3 if p.style == 'math' else 2)) + 1):
            choices.append((count, f))
    if not choices:
        return ('call', None)

    if p.style == 'random' and turn != my_index:
        if can_call and rng.random() < p.call_rate:
            return ('call', None)
        return ('bid', rng.choice(choices))

    if p.style == 'math':
        own = totals[turn]
        for count, f in choices:
            need = count - public[f-1] - hits[f-1]
            st = stats[f-1]
            ev = 2 * (1 - st.less(need)) - st.expected_shortfall(need, own)
            ev += p.risk * count / 10 + rng.gauss(0, .045)
            if ev > best_ev:
                best, best_ev = (count,f), ev
        if can_call:
            need = bid[0] - public[bid[1]-1] - hits[bid[1]-1]
            bidder_dice = totals[bidder]
            own = totals[turn]
            nrivals = max(1, sum(t>0 for t in totals)-1)
            st = stats[bid[1]-1]
            call_ev = st.expected_shortfall(need,bidder_dice) / nrivals \
                      - min(2,own) * (1 + 1/nrivals) * (1-st.less(need))
            if call_ev + p.bias > .33*best_ev:
                return ('call', None)
    else:
        if can_call:
            pf = false_prob(bid[0],bid[1])
            if pf + rng.gauss(0,p.noise) > p.threshold:
                return ('call', None)
        if not must_raise and not skip_reveal and len(hand) >= 2 and rng.random() < p.reveal_rate:
            face = 1 if 1 in hand else max(set(hand), key=hand.count)
            return ('reveal', [i for i,x in enumerate(hand) if x==face])
        for count,f in choices:
            value = (1-false_prob(count,f)) \
                    + p.boldness*(count - (bid[0] if bid else 0)) \
                    + rng.gauss(0, p.noise*.8)
            if value > best_ev:
                best, best_ev = (count,f), value
    return ('bid', best)


def _random_hidden(rng, n, preferred):
    if preferred == 0 or HISTORY_WEIGHT <= 0:
        return [rng.randint(1,6) for _ in range(n)]
    weights = [1 + HISTORY_WEIGHT if _matches(i,preferred) else 1.0
               for i in range(1,7)]
    cumulative = []
    running = 0.0
    for x in weights:
        running += x
        cumulative.append(running)
    out = []
    for _ in range(n):
        v = rng.random() * running
        out.append(1 + next(k for k, cut in enumerate(cumulative) if v < cut))
    return out


def _rollout(s, proposed, seed, preferred=None):
    """Finish exactly the current round, then evaluate its end-state neural V."""
    rng = random.Random(seed)
    you = s['you']
    nplayers = len(s['names'])
    shown = [list(d) for d in s['shown']]
    totals = [count + len(visible) for count,visible in zip(s['hidden_counts'],shown)]
    if preferred is None:
        preferred = _preferred_faces(s)
    hidden = [list(s['your_dice']) if i == you else
              _random_hidden(rng, s['hidden_counts'][i], preferred[i])
              for i in range(nplayers)]
    policies = [_InternalOpponent(rng,is_ours=(i==you)) for i in range(nplayers)]
    bid = None if s['bid'] is None else tuple(s['bid'])
    bidder = s['bidder']
    turn = you
    must_raise = s['must_raise']
    action = (proposed['action'], (proposed['count'],proposed['face'])
              if proposed['action']=='bid' else proposed.get('indices'))

    for steps in range(150):
        kind,payload = action
        if kind == 'call':
            if bid is None:
                break
            actual = sum(_matches(x,bid[1]) for h,v in zip(hidden,shown) for x in (h+v))
            if actual < bid[0]:
                loser = bidder
                totals[loser] -= min(bid[0]-actual,totals[loser])
            else:
                loser = turn
                penalty = min(2,totals[turn])
                totals[turn] -= penalty
                totals[bidder] += penalty
            starter = loser if totals[loser] > 0 else _next_alive(totals,loser)
            return _score_after_round(tuple(totals),starter,you)
        if kind == 'bid':
            bid = tuple(payload)
            bidder = turn
            must_raise = False
            turn = _next_alive(totals,turn)
        elif kind == 'reveal':
            indices = set(payload)
            selected = hidden[turn]
            shown[turn].extend(d for i,d in enumerate(selected) if i in indices)
            hidden[turn] = [rng.randint(1,6)
                            for _ in range(len(selected)-len(indices))]
            must_raise = True
        else:
            break
        action = _internal_move(hidden,shown,totals,turn,bid,bidder,must_raise,
                                policies,rng,you)
        if steps >= 140 and bid is not None:
            action = ('call',None)
    return _score_after_round(tuple(totals),turn,you)


def decide(s):
    """Return one tournament-legal action from the single-file trained bot."""
    started = time.monotonic()
    legal = s['legal_bids']
    if not legal:
        return {'action':'call'}
    quick = _QuickModel(s)
    candidates = _screen_bids(s,quick,legal)
    if not candidates:
        c,f=legal[0]
        return {'action':'bid','count':c,'face':f}

    if s['can_call']:
        call_score = _call_value_exact(s)
        candidates.append((call_score, {'action':'call'}))
    if not s['must_raise'] and s['your_dice']:
        candidates.extend(_screen_reveals(s,legal,started))

    # Every candidate is legal as emitted by s['legal_bids'] or reveal protocol.
    if time.monotonic() - started > TIME_LIMIT*.75:
        return max(candidates,key=lambda item:item[0])[1]

    preferred = _preferred_faces(s)
    rng = random.Random(random.randrange(1<<60))
    trials = []
    for heuristic, action in candidates:
        fixed = action['action']=='call'
        trials.append({'action':action, 'sum':heuristic if fixed else 0.0,
                       'sum2':heuristic*heuristic if fixed else 0.0,
                       'n':1 if fixed else 0, 'fixed':fixed, 'prior':heuristic})
    active = trials
    for count, retain in ROLLOUT_STAGES:
        seeds = [rng.randrange(1<<60) for _ in range(count)]
        for sample in seeds:
            if time.monotonic() - started > TIME_LIMIT:
                break
            for entry in active:
                if entry['fixed']:
                    continue
                if time.monotonic() - started > TIME_LIMIT:
                    break
                value = _rollout(s,entry['action'],sample,preferred)
                entry['sum'] += value
                entry['sum2'] += value*value
                entry['n'] += 1
        if retain is not None:
            def screening(e):
                if e['fixed']:
                    return e['sum']
                n = e['n']
                if not n:
                    return -math.inf
                mean = e['sum']/n
                var = max(0.0,e['sum2']/n - mean*mean)
                # Small exploration bonus in early elimination stages.
                return mean + 0.20*math.sqrt(var/max(1,n))
            active = sorted(active,key=screening,reverse=True)[:retain]
        if time.monotonic() - started > TIME_LIMIT:
            break
    # Compare only finalists; discarded ten-sample candidates have a noisy
    # upward selection bias. The analytic CALL remains eligible at all stages.
    evaluated = [entry for entry in active if entry['n'] > 0]
    evaluated.extend(entry for entry in trials
                     if entry['fixed'] and entry not in evaluated)
    if not evaluated:
        return max(candidates,key=lambda item:item[0])[1]
    return max(evaluated, key=lambda entry: entry['sum']/entry['n'])['action']
