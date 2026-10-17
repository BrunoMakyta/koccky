"""Standalone Liar's Dice bot (2-10 players).

One file, Python standard library only. Neural network from placement_mixed_60000.pt is
embedded below as compressed float32 weights. Rules follow the provided engine:
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

# Compressed weights from placement_mixed_60000.pt (23-96-96-10, float32).
# Embedded so a single .py file is sufficient, even without numpy/torch.
_WEIGHTS_B85 = (
    'c-jn?cU+EN7{+O7r@g0k+R=EQ`+h~pmXW<HNk(L^(nO1f6e=l2CE2~txgUjOltN}Qeq=PPtdO7o&mZS}KG$`w?{~~I0i9DE#5*F_2;'
    'CQcz>#Aq+yG5EEcow{=!H%@EbCna4;Ls1IVD%2lPiV9Fex0WmH>lv^}wZfCiJVU<3E`>LvX)NxWDlrJbrba^po7FY4|DbRg$S#!Fma'
    '8iFTncdD1ktT#qW}RI}lgF{HJ-Kl^JmlhiGzv5hjv_%%C=_`e0u_@yEzHf)Ttgr9$z|5gydU0mnE%{gbn|6cW$pU7@v$+>Nq^)Z+Ym'
    'vKN7aWqTaX2P()NO&|cjV9i9A;;uoYAH-1OY2&yx#~t&AI^cCZzAx1zZdK#AUMSx<bBT_=Q?@^z{P1lMfk~pA7ZG^mzC;^UvBoLLbo'
    'pbr7%b+Dq4y9d#2#ef)`k!=0vGx$z-;FEfuDXqZu3Rp?UIcc&fb~&gg7}4f>Oy(cTBTu86_ba4<y9NCb;U8AwS@g^$5k;IHx|ur40P'
    '&UWR4>b1jcTKi{A4EV`>x1_O=YX4zKYND{Ae-$l#+rWnCrBFi_p~901;M?N>mu}XB&eXkNp4tjMgB-EKaw6<HB*jd&J{1kUk`I2;e)'
    'LGqU)a3t0LDL+VemVYmA5@&<w=Lx!uK!PcZU}~)7PG0SCo|TLicT6#@mbxPI_Zl@f+AWc?uP*@x?3GpMvjPz;8=`K#1vO7^Z5C-|b!'
    'L^S&aQGH;1+t1B5V$*RyG$1L{b^>toh`EEA0Bb~9~b6M%Hi*V@WbtY>v3H?H@W4viHcF&Nf3!gL5;6)p|tvC&@TH8`Tv+X!MQI_oGv'
    'hcsmojCp1es+9Uj*zN&7p;>dSk!unzD#&X!Q5cDarGBNH+34-cauA|ehR%%Q=x3P@9gc23eHJSnc1#>$!rUg*scfvG1J1uRG74nZTf'
    'SE-+%rApTTwUr#hG7(kC{!uy&ecMf(uAvaSq&YZDtA`3BGQOc0KwcnMR(O9km;8p8EpRn$HoOlESEp>Oj(FcE!$p}Mm%Kt2UenM^^O'
    'yHA<yppP)e!5RzVUb2*0F{K>5L;bdy<5`au+NV8_zTcRPX>A8FZpC{H7#KkZrroATNA5Ap+uGuDGRM$$Od$mO-Gy@+M_`uMNH_+w;M'
    '@iqes6;|)|3&fd}@hSE+#lVL<$XtxYA=mjgEZSP6Mr`<7Xc?eB@As;^%T``7dv}Lym*$b|-e}iafep|3>d`w!o5g3|8Lg1K&tb7^!z'
    '0G>#I)g%!X_S1IhDriE=sYPr5&OUbdhhq4!yfz_{(q&{UK%`2Kf|J^QTiq`j-CghM#K>>R?%#K#RI7MP^H+@LV7roM*&+hCnX9x37v'
    'Ozbi+04m23rsMhea=t#DDFDTn6AmnSKh(G5LNN#YmJaL=MG=lvmZv6SVDo)fBelGrO-EQ0mOa%&F8wT<aMvu3Cc&Th2y=y!EgKvxa`'
    '#e(ai%fZi+lsY0062s|wcM(8R_CR;YZ-l-lfG(pQ~CHl}zC(!>&qdAm#$V!WJxwp0P1@rJ_nohCx)(^a_0`vwF^U5DjG&so%L8+^Oh'
    '1K)2E<25FQkwH^XxU(K}?knKQX|ia6_Sif~1$Aa*uyJ4Z(5t!6P*Q6k9v2Wz6_7-NUpSSF$)nk(S#%|*fHtIsQpA*GI#ydu!o44yP<'
    'Vh#4m>2$agC{77B-7>m|Dpj2X#m^2ZVB}%aVQ0bqJhoz%o=_kjaU5Z4^c}wa~rSTH+WDeOi*7!EMix!Ea(syli-m3w$Jv-DXvw?sx;'
    '-?%ad0mun!PEfLx_Er3_?AK=KCm2kj)KbX}{hg2VPV16y|HLDRi6wTSf!2v=vXD3_?849g$+F?cAN3QZlAXAsw19QS#=<>96RIXvgv'
    '>xkH&u29(Qo9Set(v&B&jQa6alrMaj(FH@7K$P@@#5D2(EM?iw;J~oy3-=q`>EAv>i2>h=#-7mqcYLUs}LJ^#Il|#;q<Gr7~TJxVwm'
    'P+;mxHzIDFxC@R4!Dj=b}ntj{$F+*wZYK4)p8_aF?p97f(VkKm^IOEh-zcluQLfKE3{qvO~otb0&4bm_-&x5RC{r^hN`_LOI=wcxa5'
    '{=EjSpIQxtO>>ecOo+oLlXjxU=}@Fp5$!u7VcS+kN_0GTV7&ZhRyio2O?Ns4Gb-cBvLjr$WV07Xw~5J7C!5|kU!gGJ1pV@MrgM{y(D'
    'X&p!l$6Cq*`w%%vqZvv~0W1ZE6pNq(LNke|jmTv$5>d%WSx+zZkyX%7hn>ibZ!52C>m&)Wo@WvzSkzDt7NFBA3MiJGZEW?NIv8dYt|'
    '6-c}j3-M*R*`8KgPMg7EfK_kSIZuH08&javAGXp1!R=D9?02fXb!IsiYiR+xf@D;tlb-x}o)md@-H?@)e(QM|bR)F`4C2W_r1C2WMn'
    '>D7Y)6|@4FzVK2&^TPhg#FT_tDj7zgOBmfm#W#7058_JR$%vy!+6c8$85IGH+E57kv;A>zz+0HWWFsWc=`HKRBzHo#Rv~!>#}Q9w{1'
    'GxbYv$Rwc;RWt>sOU*d1)eLnAU)siNqW*XZ*0fjG0XKX$9#0nfEX;F@8Eb#4qwv^qd`r5Og5euSYK8mMq940<!i;L8+a@zgsBv{-Qr'
    '(Dj`xWq=NidUOqjT6E*K`ZqYDDMIiZA3%N6MpDt3H7NURBaUfWg%RePaMYJAIACo&z8kU<ePoi+XucXH^d3a7ptW>yi=r@&EyX1#_m'
    'O7(QPlGZ!g-^tgaXM-=&O@q`(O@Otja*U@RLx!y%Z}mSFlA_j<d~s5_xMJ!tYzWhb@;L#gs%3B&}N8L?;R}*z3M#w!>i^C8QkXdWUY'
    'sr`i!L{Kj(jZPG)C`0pB<TO_a}{y(8L#f5IU)^YiU?s#8H2`WS9g0pfbTQk62QaRU+TlsnvSWi9#hAJ+2^@sueSWzolzu*S*^E9KaW'
    'zoFf1~<BUu@$8r6{B{hqL7<^ftCLJckcKtHoMQ3*16B7Re8tRx!(f2u}n#lDpSq1<Q78N=+lyopWR@8+H9_#bA{;PX6*8o!7P3INy&'
    'ps1>Wy?3$|`_WJmtaM*Tn2gbQ0mm^>*5qS797YQgCs-*OgK&2qzYp>?o(;|jhdE&(luEk_L{JN&4sk12Iq>5qFljozq4oR<@;_c=;i'
    '<MNnhmY!g6p%+FpYT=bWL!mI=mu~w_1-)5rWPE83=-rkR)C{d)=C&H9JjDg$(`=dcO<!;hc?j+kmOxy>GN>=mfWC?@m@r1dZFqMOlE'
    '-=RzELVr=^w}@{XD~KM7&1#yF<j+dt%9U^c+FQQdgLI{w!*}F+~3h(lp86hPnEM;OVF_Aay-bR5PqUTpL;g&P!6*$$3g_dZ;brY7Ud'
    'Q^$JkA(gP;dp5V85_pt7Zoy@Adp0)Qbp|<fJVqSeP)>p~l<sLP3dUF=${*1y?LIp;9juqXzyc{cMgy5H%&e(6IGE;q<#G1-td6S=Y?'
    'DmrNY}5ALtRX{<tb%v2;YH_J?7IVWeE1+rtl$LK1&^^+z6meVeKzNhF4g~yr#W@QX+&!V6)n-D`h69g`@{U|KXN1B-TGhLuBJH<b|M'
    'b43WK23#TqIjq`<0rr^qGI7$hSNvA^96+~M^Y-Oaz!=~!(_xV8h2EF}#2E`<-zw}P&-HrBV=;ar1>taIjCQNo_jKqW!^{8ffn7y1<5'
    '9Ulry)E>f2>n6~$m&W~FahOvokN1Ql%<Zu}X>^oOcd(*(aiJ~k+5MRMSAS(DKP$+r;uNiZ^PRJ|dw@!6pLpY6SxoJTF)U8aW?38Dm`'
    'z9zA8xsuKk9RlxmAhSq$$7He!KJR8YZz<Vfttutcxohx3kZ!{pie9E#@}vDA|{^(!}^`HhM@jZGUGi)X#CGs)i~G8A@zmN-&1M97jP'
    '#voPz3KCLxbLMDr5;kZw}6f+~1afh<lh%Rf~{MduWo_L2)<07bUt|Y0ODOmQP9FtvI@oecgxD=_6hYX@IdRjO9ZZ<-l9f{nEm-is_k'
    'RE<5u*6FSHE?P2EIgG|4RRgw`0+q6Jnd_Tx-&L#;}*|!^G4zP|2ipnh?KbBqXuTASjyKG=u*Bx0hNE=Me3S|=uM3x9X{_z=YJTJZuu'
    'cL$J>T1m&;R3j2y|nQJ{Hg79=;pjHc#}r(In&>}{lk6$k9X)Mw|he&|g6qL50LH@v2tf*_E8p~XJ*yn-R|7MR>K1m#uq#I=_d#dA9Y'
    'p=M?*m|RW*$vYKp*O$@UA|)xPY+nv)=Pqz^c|Sqx&>7g+F9ViUJjQt=O+e+gv=Dt=SAZH#+Mx8ARlm1o+ol;Y?-{>Y{YgU#tjgq9P%'
    '(?$Fb6(A-w7&R{n6|BS4b{>4Sg<8;qB}MxU?VvhR*PWtEOkUtaMjslF=9cICK@qcCQx<-UU#z+ZZ}q7e(GZ8!-5X4!-e^!9CR@NQ3L'
    'd6}9T(a;N>mooYR-a}}Y-Do5nQG;wa217?gGice0o!ri}v@Z>pjG}tFbW5Z)iyXi1>ck4ij&O@@<cNqe0U-C;6uYm0S7;JiMfPTkZ;'
    'jF7Wrd~V%KStNV2L(+md=UY;%aTQ}PgX)+ksj<SQbyw;Pe5tZJ!p|r!(rKn!7FVG?QWV&X*Y<y-?)mhwExoIjzrWRe3F^43JTW=;a3'
    'k+{?`!~+HdMk&$C^`9?IG{;UM4#`*7H$qJpQ3Eb!nZ8BErZN9h^Hc$x2mFQ;|TdB%3Ki5W(hRC+M+@(6K*tQWrMtc3sSZ*tRzFJ(t{'
    'zVoAVp26uIkGUfYE<@WVX?oWEpGa9Tll$Zt3HocS;14fmJ$tR$@6x|qYH$y4YEi*wcBsSt<f#<+DOr^AOb>rw2xSr1mDsJ9Y49%8l#'
    'MpDrzM5zxboRewt8?CulOaKyYgx{@0N))3lD&)$_2L7^A78}=nJ_0F0*%(Fi}+^;{vkTKJQXiW4(t(F4Bg@Lx#}%Zbbp2{BhJ98I*c'
    'QIChRZzaV=A%IZX;T=XEkJ4F$d6I9r<vJ_7GUKl)F)Cy~(M#I^XaPVz70pr#iLYCTl-gnwc7`&_k*y;h~w55b>rK;F!i)rGNyBZL-C'
    'z17-CUJpQL-|v=Us#<{-Non6R`RMBWtipRDCTc+oAqW#F=+K<38{5_@t|e=z>WR6eV_L6VR1?<sqh-l8_I#Z-E4g1_=($9Fq(>8(kL'
    'Rrl%8owp<2#G?7wXS77feAoi_J4?Wc)2D_srZJc4lC*a`Ty)C(VM@<Gqp<55fB9|!Ur`fW79eNr60a8*X7eU{jxb%<J)?WVdBTd23Y'
    'g-+I7VJ;yDDfwIi-0pRRp!i-WU+RwgZgHITI$0q+##8*mZwMA{9*l*PCt-z_3Vy!ig0Gh=;bE~Wx<)wT8u>ODey#+RQueZHr%z<Rwi'
    'H`C?5X&u0xtaOfVI8*Ve@`z)RQUT4&+A*-H)~iyA+QK568umQjR`8xMqZN=MVa~P1@+%sfCLUhhX_(D_Xii0nMK7W*V6n!QS#9Gd}4'
    'eI4<o#)1%wCy8T%!Y*`(YUAsym&KyIck)}H|Qtzt8V#ViL!sq-xT5x?B+P)IufH)6y+?jw!v?id#Ls_f|`bKs79=O@bfvpc8kFp70X'
    '#CYEa$b5B_hx;eI@>eYclSG_<UInDV{%leH<H`mI1taojKms^2h=!8S$x*;GhFxh2-k}T;O#FiI3Ekxe^$eB?dD#1p!ET!Y>ftmKPK'
    '4ojAxA@CyAfk&JugZaqDm-X54xMr`JD&0>^_eS>6z}LL(&H*(mPC041>JU1<HJK@@tfj&oXan3?ykkVv(Avmft%ah@S^$i>97mJ9>5'
    'd2Na%Xi5cUrWBTFMbCC=3Cr$rbnmA(b+#X4_S+4Go-K3OKk=ZY%9d2KWf`%d;ncR?1HTN{!NoWAkv(unvz?pq?LQVbM`yI3ppCn}IH'
    '8HGKZb@z<NKGlAnpVwb};;er=K67*I#vL{2(PZ-1aM5=^w;RtM;e;U2}Pze{4QYn}H}HhRFoS@&juuxRw)Jx%l?ioS(f7Tr88~)`*n'
    '&QPLy$-4`5K@b)&$I@dsv6|HEj;Dt%24pQ@0C$^_y0cklcpx)mLX_veL&25hW|5>rP%F|ltdhCIRtMqV$nGJfh4M2me8i=(}!{d4V('
    'SQ@+d(0{L@~WAa?kpe^d!AN~*A>T@gLwJUFpBU>knDYPr8>dMh9b5~)05$j+^a)&^kB*c?!z%no*$vX`ofzyKN%O+^vRHIy|;(mJ9>'
    'mAtL)(yoA&Sz+nzGTc2hRyRX#R~C*jtPININ0LY8%dDSp*$;63kh8=lX=2WlDu-EYQC8?y03$S$FNfv&JiV;R`UwZh@h8n74`1;s)M'
    'NOwPlsx>#kpsEv6HUEP-Hu<0}1X2(LlghZA_~n|pc>X`nkx#kJt}pO|%#D{>i?$3&PTSJtr=GOmu^+v4TT4ah(d=j@GSw@IEV9y@=3'
    'l+aY=#B1v3I07DY?Ovk?qVB;^awr;tcGI<54}q2ybm4D(Im!o{BZY$>H-TKsFhVD5t>PLDIN-MKQ&0a;5C!W7!ac^%8^m#r%L~FYfj'
    '2GA>L`ky}4#6R#Uy!=3m#4i+4p%rz~IgtKyKXd0#>$p{I-y#A)*@Qgu{kEYfnTd<1;C+=oumbnxZpi6#fL#Sq}2c2q2q;a$S;O?w*o'
    'bEVJ7(RIk^iNt0vsVSetdwZ(^0pxmYLW>bj0c0mOmEB^z6T4M4&Z9^7TX55(|;GrVeS=K%4itLv>c<U*=#j$I3|;Qk!X`&cPy<pJH?'
    'lq=R;SgEwuM!LzH(N{Bqs`<Bw>;(D9E%A>(4;<urfDIPFJIHha=7+kDz+(~7wzJ9uvc9Xh%Auw>x$Gn{YUe73M@J6y<W=R2qtel$9>'
    'KMq}7fwe5(m@|j(&}B4wQVf46U51~rO&2!GWpNwY@-Dhgv*2`!t;neI2@b8@FPwe)AC14hnfh(ZU=MwOUHNN)?}GwSGEYOS$J|lHbr'
    'KBT<P1+<?}q5tWe~G552idk2d<nIc)3MD)wdSzucC%fada~)DtjZ+nRWx#mHic&wtgW|hlWr&>m9R{4HA526|kf~0;qb)(nQa4HYCr'
    '4)bbPA<oG1EwV{(+`Q8gsx2oXMH$V78yBCWT!4o7@z{x+Z=luQkxX*RKb?y$pfk6|ws6XoBrsOOf`*AFO_*p`Rv*TeMClYS1drUhF3'
    '+dJgWns!%P2v4WU6`Mq1pocA1Gyo#*f}^E($dXP@q#~;Wd=g#JO}I)h%bB_2j2HR>8yz*y?$qi8;2PQ$Bzvao_3sonPX-_PV-xd@V1'
    '_73(cmsr$w}W^>7+9>l<agG8DvR^U=D9gYR?ZL(BkzwoP@QvBVg^94v##u?iS4sugx^kLNx2WS~O3j`+H6uz05VEUMHRf)(A;_{D!A'
    'v>939<``!fHZ+9|@cG6!4!;TCemX%(eiVEQkA|l=PD4vh4EzbphTCuJVb_x|ct18CzTOOnD+_X{P1TTcH(LmfPaVW_^`jslU>l@mo#'
    'G$+*)#hgIxKRO8DEn&iu}e0QO~PQ^ys}5owb_|dh*fy*jwokR4|3*xzA>O|7p<;VFEi6@`FV!b*CE@#rUqe1dYrY^T{+~Y4Nj|1P<^'
    'av2tL2<pn>}Jb)kYb)IC?Q%6u@3GBy{P9`-`50n?lu@mX%87~_M_l{_CE=yEdxyKde74OTJISeL;5f%{b(8$k>zY2z%_4p6EW5Md<8'
    '7^YMX_1Znb+$Nd2KTq|F-y$zCG(b%?B$z_bll)AJ-krPLWX(M(C5e4vQcN)eP3;gT_(*I$`Jdw_BOjIxG>k~)l5<UG~OD25L=8i#Ix'
    'F+V4apC!csBG&N8FlJ#u9G{wlS184Jk==aEZUrzmIrR`&LC7q9%hn_afEV!d{yylA@_8+NZB4AE?3Gr7CW>8&H%r5ewz9axN3HFNof'
    '4K=u`SR0J)snHo*bF!})M>}VX@J>nn2yFvxpg`S`I}Ll`o2ZXzHXrAv`i)^B`|H`#-`0plMl>(Ii~Bstjjr@Okqovw!*;KUgMtSi*('
    'bfv&=NYBf~2g(1N#P1Q(ziD!^oOV=#%H?H79WOvZ0(w<OKF<LO&AUX=VA_YuWJ$eSFDCJ=Wr##*U7QVjkJ~%yRA)rnV-5zdAC5&3}B'
    'Gl}yS<6;z<e?Sa_spGM`z$*}rGE1bA#flCS<QR#{ke!H*~4Z0TLHQhZp#_29y+pru%yav+P+iOwp_jY`37=qTC|9mB9DW2L9j|vqrX'
    'm4jN(UOy*^_lyGQMc!#oQbb^iv9^yynF&@HEqRjwFfY(stgm>NRYdkg!Vh4aYgVU%)iqQPXrdhDf?U~uFc|tPM1RdQXYQl-i1R;pF#'
    'SeOi=Nzfvr;IG^R~O2x*L?Gxcxr$%XlN`<kLK1Dt7K<SJ^LtxIo&BDNvJii}s-kn>e_YM8FVa^J3Do=UN7n&uBCr9Y7^^oV3(KK)s@'
    'S1)f~pUydkPxKL0ZNynO5noO^fE%}}3!6r)q#bIlyyU(q=iFUM-&zKWbrqxV^_nj%VnR4SI{gzJ3mL(T<X&_8oKhrfFKLNB?S09&WZ'
    'N@wR1x=fb+G7db5C`MMFqF&`*B#bWh$<oF&_gz_Oi_n%&0S3j*H$h3<sqS<!YDj;fLfa(EF$Ltk}SaeElp?OG$?PG;*i3+H&ss$mN`'
    '(XFb3BTN+He|Bv(Hk>VW<Nv+I~UnmjLy8&MGP}&`94K2}Yvnu@@?n}@6SJT`*hWsp-XMC^KS$f!7Nl{B<Y1|n-awvD?FT~Dc4`Ldak'
    'Lm$-<4ZqEiVtTQ6Rz{;Rb!ad%`<Gk2UWIxgbY3o(iMkUIN~MqkCgF@v9(oAaJgR+3z~2ZTy}fl6*d`etIhz+@g0&e!Vh>>LlkV{fTu'
    '%7;1HV@=<9j_j!iM}+ohJ1-P8n&vvuiK;4xx#nV>fy7b|ah2};%Vq`lt)%Q8JN$M76|oDo7wBLZ+|n<37O+zc_b&+*9)4dKWOQ`}JD'
    'iyt-^;VQWRywz@xgS2#!`_BN^8~WhXqS2`9p^PEpc1d&&ey0P>5>J~m4E%YA*R}LwH%BUhY=*34fpap9zaNXf`VJIhs)_#nHlW9ZSu'
    'D`$2s`X<#CjLbV#^)Rut^1a?9f$1$)_cz<Zr3OH4kcFVOvI_)R0E-@(&Wct->koZXvhuM>>wIwV_96;&J(*P`s2T2eo6K!KyK;*mzV'
    '5pZ)#`<woP+)aP$d`C$_TRXd~V)`4h~UkwL#b-**}=P)I$0tWmkB^RIhbXe7fu8cc^zJet=X>FjAKg%gdy^wMr=1@bTga#HpB>&t?6'
    '!>sAiG7Ci9UCk8S1)S$w`XUwjv@nCc0rvbpS0pWy2-L_E(N0cp9h)BRTpvXl_YxpX(3E{w~<a4oMxPU936Iz<nz4l<DP!UNOPl%Sh2'
    'l~9+x<X_ZKS&(N{;{;GKbZdABnf$h+a*&E{w~OcMj1+hEgVS^V?wZQqLYQ9kr8TNQDfR{JKw>FvF!S7k5spX(?bo?<V|zT_dCGBFa`'
    '%B_Wt4<^Fb=D~t#l!~z1-y8=TXyU%@N~jcj3#^CVhga_m&|h5+8{@U`_DXx?((B>#;Pu@7K^mwM>?}N(94cOVax)$^l!ubbdm$roBu'
    'p`W$4`|><gOJwfETe_LD}>c1SlJG(_);sl!X7d*f4u;%uWO5nC&i7P-`RW+VxyQrZwoefAP6sWJMjLm$AA8T>-Uqgqx}IxMr0xYuc3'
    'xd3_1gRN{pTS6A_QpSrlkZATgT%-~*6RfAWz-*b-csl0WClW2}i9M_Z`4V#jb_{C5J7M5x-HCe(bXzXFfw;zU*5pgV#k7Jj{$Wpdz0'
    'zHz>Vh+v=xW5ZEg@FT#>72?7QLeqc@LNKxL-r(FU~z<(87WKWd(^2=qm^adki&b^3@B*CRrcNMGc=|B+rgNF81c#rpAKq5n^Z%wj9)'
    'C?o|q1ex*@Q;h=+*x>Ud{-8P|8ll5<S5rhqN-WRh`=ty-c=!`oWeBUhepY%ujXU=_i$|Ju08F3Y$hOWv~^Kh^-;Fe3dS-e}g-$xbBC'
    'V=sr@#dF71gdYiM_~Y7PI-BXoZ{JuBR~|lK7QK)0+@k9U8E>Jj;-51sM4@~CW>DS|2Wv?Tb1p~;-?c}=$%#sM&rcp5m%L%tm)^30d$'
    'sAMjFC8Y(HEAM{fo7&ED?R|c4uQ>KV-=v!|7SHG1qe6oci>n=)ve;{P|5^_|@A6u{XN$Y<4!!`9~b*W5ciVc_T)E_dahH7NS6BRMXK'
    'eEFa5^V>zLI0>#|=%#C($;O@*?hiuawNm27?+;=7n3;(kaA8U^jtQQQWD;H<u)d6}q=3f_m_qkzv7@|_?Fq|soj}88&xYT+dTl#tyw'
    'vA5_dHnPda?2L8lQ&0V=enPed*KTm{2PfoQ#~*(eI|Z0NJ12!!~mNr=rMkaI%N%zJY52dqAgIV{sn+aGkE+p!5cT9K|}asSf~9H4qB'
    'dt`yVSwR;7o=J?u{d&g{ieEAm<I#%ajLeiC^-Ng(wDtvm_|{AjuH;>|;sQ*lHV^=(uVG+&*9t&22q`FcM(^!f_)C^=!LiWBO4?!t1{'
    '3`$!N#c8Uv&|b@1<Y8upFB1lcbN+3{aj_;lyL$|Ix~-r?%0Rg$?o?CdNiRmJk@gQQ;+#&himn8v)fmkBJ0}3#QmkHEcZ**g5x`z7U&'
    'PYg!r1q!ne4GOkNvHF!J2~?@h&%zt1~dAbNZIt@t-1;HX49`l+U5fmvqpLE8(x^B=Sjr#eDRXIDA<5lYf$Y3$yvjm=tu5E7`4ufAbH'
    '5rs)$98n&UTNE=5S7U8zN$B1rdV0dN|xOt35*58~~&AZ6nnc9jJ?;OU>f6PF6i3L@IA=7bPfr00v$WBWK$2b;&<HBrK(<ZQx4@%ti!'
    'GoY+cn#|9U4`#E$|+A>hJGK<ftdLjq|=s!k3Ys>$f2pYIldOScn!3>KZx$-RfvWf%hKMzjd*?ZENYkCjP7@4kmN}&I5j3wtNaR_k>i'
    'RQ7Si}GWetK)Gg>Wp52YGW_(SdxX?&VPk2;UgvWf3lzNRPXZ%C)b(hX3a_L<Glwx>>ab57*g#a8b5!5#3{f`Pk5crMZ%HdtN3&pY<B'
    '*no5xJKq})6!>wj?+)Pao>abKlOg%vVJxUCfudu_!^w-zxXH$X`K*m07gG;5_v|IQ7&`^`pV&$7UJMdHoMt7C3%wv-m#ayQM^QZO>~'
    'VB87Vx5CJ<Tc&$KdG=Sa9DLw<j8L<(w5-IHyqTqWNNF+KY45lTo^gNdDbeYI(U(e17~$@^G?a{X%Lfe$shv=i7WX)wLd}?ISE~%c64'
    'mPfYn>6P<`T!LH+8Qu4n{Nzxj0Pyau*JF*hxI!i^X%)9aZY<IzQh!MZneH`k~Zlc6gLp;8Alz3Tlp;-5QAHEOt$KY*~xwXq)z~`Aq5'
    'ya`(LLW)vZzbdx3>MtNP06G0EtPL7z=_++>6PhPw%g?ZemUHR<-=65%KjKkDH|uem|eyq2ips;haP6u!xoAXbyidH1<hf(Z+xkLtTL'
    'S~ctA%|j`5*EN<!EAry_+1*U_;oliNQ(31QPTlo^gV?anTt0ltv8-4sd@71HOlF+ybWP8jmx4;{|-7Ejj6!sD05;EuNo$ib<dOl%ui'
    'QerRV7;B*Gy~!-GZap5}FjYLUIfms}2hzJ*gemf$xYL&(^M6_{!9g2$RO`$VlAqpz`&Xq|NJlQD<+NkE>?Nix8;)^*+{tO!6k6Il2)'
    '$%fVEOMN&^=g0)BY6F7nMel%D=y%(`G!|Q+R^$mE}-8?kGykEQEgHBXG@jIbq`V+tA$1WA4*zJl`UR?-l*oi4kht^?;WQzubn*sS#w'
    'YYLEB4PGZH06<k^BXtADhJh~UBfM56%{%G1XEZBC9p>{ZZtnEXENCUdJ;}`gaI0&Cy@3M{4PvDTZqtJWkFy6tbhF)8C&<%*Dq@qEjS'
    '#t=wRK3aU(03MKT+1|57E((_i&)BUHQPL+jqY8OhMa3t;iq^zovWw>SH;wSS1QG@8KZn|&3VcC$wp)4vZKQLmkPqFrA450R7O}XG3B'
    'fS7GPwLJoDToBlaKH4FmU8qHIAJyZfIVZTQs4#-~>>Pe~p7HM<5)o^0di7M9>Cdk5CPat^LFbcg1R?`Yx|S<3BO1k3hzL*JG(+VfXT'
    '@fmB$bWRC<Utog|D|<x_?<_#1l7%NL^GU5lO&ob_4URpO%U<>~V!jVvU#vZ_hP&{hzqrP2IXcc9N=Ek{(+Hm_XfmZ4D;5%+-Xx};=e'
    'yW5t;2%fjZ`)`*^BdeD8UDcC-C#*?d;qk5v?1ig>RLv!P;MH;=XCJxOW;SUguUpQoRQJpIfI{ux}b3(Qv`mAqlidZ6SS~^nyLN-6bq'
    's(eRHK$AeVY{fkrdQdyns0DOGRhKyB`F=hEgnjW7|KVCU8%Uz1JerzpsqB3|hM-7jkb4NpwfjH1(6;|#K<aB?Q!Yb=%nj>FLyEL=dP'
    'Qxz#O06HMw{PU7|F*)trgkdLTMWZr`wH(0_H&~e2a}KOM0mSEj^#NxP~78G{4gby61N|PJ;8ynL3r%rW$=(29<WZ*TpI=UvYxPT_b@'
    'mlYsJpYIm+C2k7l3N9N-E{r-J|5v23$ZB=|a6RZkdwRN~(K%sXn94ttTc4|*=G;2*D609l#q%ydl>+dHpQ5^J{}X4Y`9Y|CSbg7g(O'
    '>A_BZ!HI)RO63ZZU1um+bS9H|PuL3OzoXbPzrWmvU;QE3@G!eORuk$sOyg7Cbzs|;SV&3_V#m)quqly;V6<+vWNB@OWY6f);JbVyGk'
    'P<LpX9k1HvKh#{Zg^)uKWQ`_fsva{ZquIZ?FaJ!rAQF;#3y8z*bbXSe^U5<p8_6IE&3Gf6jJYKJNYfV*)$ACzqGI=EHJRquHWfFIJp'
    ';h0Pi;h4r6O%nN%;*{%EK%;EPiI6GO6zZBxkydKH1)duNcd-EF8ve?g9RTay+mcV;Vn=NTb$ze<E0zm1<C6=fX4_B3su~DIO`7`rYi'
    'n_*aU<<D2!1Py%j0=9uFMSh7H~&th;2}<!=k<$I&2GcPsr|)|9{7l7DH(&)gcx{}X@!0T>bUKy4vjKZ5N|NJg4h0zWeFJz*<-~F*1T'
    '~Y89v~|i`q|uN5UFBdGjOLnvBOGrzeZ&Uaw-6+E*d|@Bmbn>86CgM{r(PAAY^~gp5lIC36Sl(cZQMy59YR+p)HRE`FU)q0$3|jD-y}'
    'J8-4wme*6*8TLowz5WNqsr7O7QqO45uKsjBbvWBT_%d`xtP$_KUyb^A_TtJ`5ucEf&+<%`)3}TV96fO#Wop}t&&*gvgJ+r1-98;09a'
    '4sp-h8gj$&KFGp9K>`8SyRS7oy#te6ThCEWLzE(DvC(uaERY#bXBi%Ghx<=G6xA%h92N?gBZ<yB9U2dM=TLuN(z=a!csgz?Hak@m5M'
    'mHUVdi;Z$@XAJbHO>A3MFu3z|1=Fsej4<}Eh@K{};HMER^>=ST$dpX_F(Is~-i#Femp}i{aB-7z7$E#_ft#csGySyA%we$StC%f6{R'
    '}xg4x(siY9LF1-HoWHPN0|C@Djs<jO#`L+!8X~A7#J!GmIkWg6YU<jE!CbhLU)maD4ExLHd1`4t&Lu6+01>ncm(77x|wNCKR)j09_+'
    'WXi;Y%|1o6m`G;4AT9lf!E@4Ds83qFd>!Pr%}GPRl0+bN4Hx-W9`Tjhn=?6Fk;TS*eQzCXIzZKbwfIq0S803&}rq`!*}Qca-^dNs&!'
    'j(?`mn%&XW*`=dV*6advQz~E$ci%9pOhfuGV+ErEJDfDngj7~$)1$04v}&I_)2cYgy5=eUa{&da95k6N?$xI8dSc1F(CN^sv=B;CL('
    's51j<)4Yrq=g2pvt(Ijh6cl_Vh*JP|<AGWgt%;Bj#byU{zRK=)txf^$?cpWr3^5O3<DV%Oy_;f&oqM`CV{;tCIBRtu319u(u<V_Dm#'
    ')+1J7Hwl5kjQ?8E7tYr5F_~Jblf@2Je>H4(@3YlReoa{7YwRO%EsA|tHImiKj@G-qQb`)|STkzdK&G{T`WSNpQm>G47=?xhN*FV@pR'
    '$DmjeX$wyUT$Sgr~dE(qdZVatCat3V#Q5Q&4WG9ZQw@4GV+-IlAk0QNc=txxO*|1P47@d>thKps#J^eRK{~jE-G9a8wmDmi{YH37$j'
    'fJ=(39&DhW<BY1;@;Xx|7KdY9ou^){CIeHXtj;yO(4-Ho1q<$b1^e}NOO=b@+F1#eAtq*AZhIPVaH%SAb;J-3CI-BtvWx=;)WwjeL2'
    'f&;u(Vq)?Tm|T1l)=d;~MZsZg@{?BhR#igP2M)3|UzT8J^#xHtP&ixRv4mSYa}jO$u%AuuJk9To{K*x!*TAC>wlpDerr7sgB@B;Q$t'
    '_O_U`G}{;HDf)gb{g{pz&xOTYNB=&9uoQQ{^34z3?OpSt(6(X4#|mHjY+oG{SS$HZbwjG{Ng4fK=xSynodXr+-W)qoL=Z$oU38&rlg'
    'X{Kk@F#7sI>wuJBeQbiSC8&PwtA^W^howT%UQE!Hgc%q~i#&8O#+K>U$!@{@;iB%Xg?KZB|T1H7pn`o;iVft1LD(+GjWG$t|(+8!ni'
    '-%5tTk8N4F`nXsPY~|^#A=h2p;=QF|4g~fEy?M}&Njr*=30B!Xy%JM_TL~|BV!?EnGU9xhjN#zum8(VAl;nggcBYOCjQ4744$<N?AB'
    'T1Seff6Ax%)vctpX!8$`mFvv{>WUo>;I9H!--AYbF#&}s67-Cuhg$K6m6HN;)!3g<eadQPgK=lO@S`(6P!p23_T6>gNP9`MJN*^|ZI'
    'c<R4cdNO`0>6SfVqw??3GL?lw`vxhYMoW*4dp8m;3+m)u(jsy(eZgw393btjuGDc}Te#aCik}yo<D8HtxR~aIw(<_bgqf){F>V(r2e'
    '^nM6LVPg)?~Whwv%*@r{FA^D^xmhH+EWv3L|(!@fhnebbf9mWGj6^1G2=a{mS?!F;!GJDuOaq)(F4NGU#@{0fLkBA~-yCIVwGxgW$J'
    'P*b;RIZ~7?Vvn&hY-2SiZSKJB=v3bYVEBnx~eeK+@PFrF41~q)}Dnw|qN~LbubT&FQ6q~2K;yXuMF#i5GtO(i3b}dmCUjI?11^HWqP'
    'ZwU0mzooAsU;EyEIGzr{mzBp1(WgZgdxoQNHPETqo1JnJsF2htH9R{sTjYY?H`|y6?C4j#wHJAVP}_>*m~M-?4MW4&Mwdt{>V7cx9{'
    '`VTA%yyVcr07kIO!BeR>vqs@Y%I{P7giy}yV085OZ5H8DcFV-}tc?_}SaWW?p(1F6Vrir69EOMGu#E?EpxBk9+Zv53UN?xc16gKNp`'
    'L)TK87NR3I`AV$+)ARVKA0rL<JbGV~#QO6k_;Q{y&Z@T(ANz2aBJO=-di^)z{*_m;YqYIUojVAVDm7ry%Ljsb@=nSRmxj0tFWB$d4A'
    'X*!3;S*F!P=9NG}r1XdptNteCypzoME>Vs@(68&o~t<ow5hirGH8WM@r)ap^$A%I>BUHZ?JNwaG|{8J5!9QCKaxkpE&P<@Yn2LcE>n'
    'iTw(^cIt``tPe<YW)(Y6WV*~_ElEaB348<CQ$BHj?x1&?638u@5c;!_i@zsZ4|Mu!9aCc;dwOKcD<I<N@*Y686ZfNA}6IDd2QO<${H'
    '0Z{+XnwqhK&|#RZ1IXNsM6X))<af9>-Jj7+dyr~JLbshKWO4lP1#{9#xtbeYPgei8q%EH;K^VUVTj}ut*Ll~e<qjEopV<3NPVNwdVM'
    'oYcp!aA>h)PBmUm^39#whQ)$d@Ydme(3=~dcmSBLKcHe%hULY%(yJNMyi1_a2jN1H$S*bx4UZ%ZCPuZH?V&X80p_R*tfEy<+RY)>xh'
    'azp|08(2@ZBj<SJ9C!A98)~JLa9@WGfQ#%6pY_p>f~=fCVdZlAJyQ#sjQ(->ZVh@;n+>bKn!$~UcVM}_I*IF@sp4cj!I=w8;nX4iWz'
    '-3NOxz3h_<cRy{8qybKka}yR>x_vW(bX2dyiAUpg<pPF6GUK-ei+LP32cbDY4bri;<uDkaKad!owjR)Lr*4Psi3nP>+bUEw#iXfy1='
    'C`H=g&5vCuzz<hKR(X;j|3x45>-&~EsPVOVy|MD39jIyNvCO?9^nNQ(h|9G6+r;0^3`fPjbT{x(Ik*(F~<-$i>!>^VlG}*QVQnMUzd'
    'b2#+JH3Y6w)Z16FQ3G=YJ^JqmIjHo#NXiR^t{k0{RoD>zbRHW>wwqm47t5r4%;`U6#BU((+tx+D4Jgi)9(y|i4(%X;z}SUZ)vVxr1g'
    '!>*_y;jJWopa+`X*fm?|6(G3U~6uVarhq~P$NVcd=he!SF&ZEVEo0(K(zJR4}I4GXeYad92T*yXBCFh5ET8V@_NHRqh6@JKM5aiB!h'
    'q<R*-H%OT6@R4lCUL*GOL^#*B<{}I3&0rTUhw>OagIzAU#OIwmAUb%pgY&-=%6!LN;(z;{^QkLs<jx#-g@q4tm}=f;b}q4o8$2wPQQ'
    'A(<Z~sA&SBV+p5&~I)?_1uZ`yktT=m~pusv7j%JDA@0M{G&fY3BR&1|RiXhtI!vo^70C%%*k@WrOdHhxN_xxm$`FuyoK3Ui+dWG^(6'
    'pdmPe41%Ve>OKA?Mzw2eF(I8Q^FJyMRts!|zxkPrxYIs&~lTqeG7`!-^?dv$mYAu{$`bQNuXWk9=ySkIbrAM$=N1t#}#uYH5*aAMkw'
    'qURK2D72hUbAiSA}D{|#8f*k^C9&QnZ}qC>`>@>$({R~*oy4ibUDKVbJRVUs<EC}`D7cKs2md?mCj``S2e}2DznkfKN5@#-Gv#6Rp9'
    '$$9UgzEE(A`VgxLpUV3plIFdAqeF4>?j<owP?)ph4+NnjfdwT>2Vp7<|UKOC6WsZ(S-Di-x~a<H}N4+iRNWmby@V%_I#dinb;)x6?q'
    '-=qxM>zzm023tWDIKf$dir82xm6N3Z>#LDE=A#xT75j*FcbW5nAupKQ_89RiwOsuCNM5|XiLh`;8DnQWalEyaxOmGetQdQZ4)i5p^4'
    '-H^o1R8rT$fU1=SleTOHc4z8&A>8q`6(+Mp4!F0$4cefn?F(*Q9VUi}fpcOcev(f`xJ__fFmf`mOsRb~xWp{QAo;=HGLM53-Gi;q~U'
    'iI>#XtVG}Dps_Q~0Tl@1D&HaStHhJ-@;hDnZEhd<tDobnRU1<M>JQ}C66)Vh}$hUmDSSD0Myscsz{5R$z=-Zya6MGj?fBhS{%QBxHB'
    '&`+y-5)1z&O0K0XrRdWDdjkJ(;La1_Ae;uSWZ+{!%lxFfiiPxabv$&Doiq^jy-3<(P5YPV)03=XjqKf*JubaYk!a=F+zOEb&~LS_H$'
    'nQ%My|uoJH16Zd8>cp#FnB+(VN%+VgKWcPx6sD+D_UGoK^~TDw(AW%?MHs5l;WZ8jE<Ta-&pl3I)&sV#gR`;`y;WyLOR@0K_ZX0&3U'
    'vRHayrT9rdO>tzCA3fSXo5ecm;nV>~(bq&kIY(RCIZhcYJI#d)!f<-9y^i>DeQXan%tjQaP}|9Ul3~&}@aBUcN|=k()2=LJEw5u8@s'
    'UDrOCy+trg9#yf~Zf*ci62u1;K>xW;-%xlboGDd=)j4<Bo$g=|Tqv4XuQ;Lx>Lj?Zw=!E8){%RpH~&Kg{i^j1W3m9Zo*95qwr%hsbl'
    '+$sw|zxOIIad<u~j?~>eN_7g^v;;c;Ar}_}J+|nWcXamiC=0Pi}(y{t?ASRgBvcy6I?$HD#GWsPA(;hX_<9^3cKdKOG7yO~FXfssQ`'
    '^@}(OL11wJ2ca-X6q_mgZ9unxb&bZjX1Mf*!XERJ9nj8ax=SwU&0T;^6lB6krW`P^*p8Z|D1R6>y?Oa%~1c}rdd311KFoMS=m-_VXe'
    'C<#LPi>x$HcAI-x=Xr#|MalOM3fEyx)z*eC>CF~q^S51Ie_RS=&)n&#@&vat`EaeTumoTw9y{Q?wm&B`u*^@~I2@8|ij;#*qy^2{}8'
    'QF0SDmtKZL1s=G5VIoerx`v$%O`@~+2GSb4Owrvz=h1Vj7OLIPgj=r;V`0}fYHu7)>SKaXFMABOrj)^16E`{*r!46F^M9>5PPDPsPW'
    'bQ41abGVVJv9dJF)&vJ3-a*4DFtH2=kV#rijKH@MA+I+p)tRr{~xRpPpHxm&-ytpnn{tO9#?WKT~XfwF8Z!{y<H^6e!j;6+fQ@<Z_o'
    'm@W3_jqIfK>D@lNmLj}maHN`sp6rsy?AfKkV6VqH|L>;Y&zi(x6lgt&dTOp4XYF>jGXNz&hAUiz4cCd**9zu?k1HNpP1<l<1Ox3)Y6'
    '=N~3v|kNs6U1~QMU{?T)|b@hA7VWzMd0ya4`pg5p!-)()G&%d-KiOLtV0SeyRByGyY+;v1^I9<tq$8BNQ(~~&BT@IfiUUPQD!)z4$r'
    ';`rml>^C~Lrx?UCttVu1l!JYuk_>=qrm+QzLIQ9$#snoE^-#8IR!o~W|}2agRRN4}gz{qbjm+Qzb<!WGC)xy`+D{7lCqb;Xl{1*&K!'
    'e$B8wyz-}9UM#<iJ^Ld~N%!h7wZfg%myKu1FH7jD^f`=Myj@uO?KNej0d6@QfCsI0>7&IhSidTfHnlLkm(~UrwRLcJ%L%HAf5*y}4C'
    'idWssLM&!`x{vx1^v~<gNLeDOyitTU;GjZACqMKSCOs+*07sz&7r|x3`>5{s7{f3Vpgh+~XDB-e8Ri+j#$-j9U?>$lP9paT4!rF1=|'
    'q-2X6ut+ZY&8Zf+<5AN?N@%ZmFD+}~zlfNEeHx+rN+IWE-yJiBBLy}mj*(Kh=Za4pNYbrGEisP0f-(tsRPJpH|8QNS~BazcM3qf%*('
    '2Sdz=CC#Fg;6s1u5T60tVm`l^KJfhFoN~^XK~ZUB}lTz-RAlXqu3zPQx<VpU>^e8nbBJtHZt`R7x$urnVNQs9v<jV3dN<&O#eI^HgE'
    '=8H#3VJvX10>j;es6pB>!I`pM;nJ?FXl3%ve;InY;rj@dkSgQQ+!vr_WetTSnxe{VW7d30YgAuk86E}6?64)car|M|0s{x$p`=g~}V'
    'h7B_un#ug-=CR1|aOm;MWCJ%T^F3XbkQ6fpMm#!AXLf3Gvt5l*?^Fc$!*zgo#*ou^;j=TnN+=~+A)E0-qxh~BO-yCJ0kp|z(XB=6d9'
    '}eW@Wp*u{M(+!oTXmF))igMhl>+^bc?}XY0}^~@*+*`*~DH>JBM7)0j6$biDga7bm8wpQB1`n_-vSr`_JZKms=@Ks4%0a(;7kg_edz'
    'Di%{79lo^D5;a+R^vJ<y1u)0gCg4z4iKKCpW;q!LJYjE!9apE%d7Ut2gvEB5+Z5zziISb#Zkd1aTq4%-jRHGe1lXZVW>7-0Bv-I+`x'
    'ADXs(IzPJbc2si`_ZKX2_*G#BW+vaz%Hh`(39hTS!1dOE`49X>`qIwj*&j1A}J&8>}^fT>yDw%@j0yfrUQQ0tzx6%yxB>MXlM^V!M('
    '~?=Qj(_*oUSTxKwkSty!1>hS~Mv>)waCwLb+eV&Z6K-hCd7KNe%Z<lo$kkTY~Jsh&M>3#Oo?Cve!(nc|J-(fmkFRDEJ0+VEr+E}A?A'
    'SG;P$g?5EZ!Hn?j_im8HoW_)uR+vY2;#&1b=;rjDj=xl+z87J%Ek6Lv$Bh!VacR&Xt3|s9%F`D4T70K(j=u{Zvy)?T*oOXtxV*yGB('
    'xpG4S#fy9Y`mKs@HHZ{S`O!qCGw~u%P4_&NM!@3BDEVrzi7<(7f15=r}u!_uLSH`<j>Gr3r@YW}*VWM}Ie!o(ZGBGoFHhc^N04H=2('
    '3=JG2hdvX`6qR?f|IGU&X62lxyNl>@LaRKK1#8xR%+kF(*L?-jcEf3SRH+!kIITc!;mT{f?_Ms1q<;+cQvaz3hSn9G*-0b{BXwh5_O'
    '!Wx+ylonq*Juh6J{FSaU;NNuf*MPl=gl`Oy#Jr!^6(3>3&U_rg{IOTnr2JUde8GFS}HQiD5J7M(;m^1wx))Lgd(M-sP{Z?LTD&SG?e'
    '+-$sUEzUvPfsKIeJv>#EOmTLtGgH{zaVGt4;ohEs063qJJ<{CD*tj3jHVuRbmUjY+HMzL1Bgesec9;(u^z5{_6u5>0gHc5$TRCRQA@'
    'B8BHVKyGIleb<P1JL>_O9uwj#CEQ8UQ=a~@TL}w*;-d1oQ2wA1r<@aKtsL~o$(LhzVa8er{j`?%T4TaD-Wq~;vm)p_EdUuC5!R3IrT'
    'djUP(0}mr*~c(v%Es^vA`y%QH`OinKGQ9XA5u5y)bXwY^Ee*Gl*(-V0LX19C#B)<==|nA>$m{`DQm7s@)*I;n%FicAC@MeQ$B)<K2w'
    'Y#Ob_7NeDR7eC)oG#msH&#M{zE#AlNYo3SwgE-=?F2j1Z6iHfyY*K7h2Yt*3Da4O`7=-`(M2R?C;1J>CZutF-QxvbjV@Yq&`m;EEoJ'
    'rgpg0lUVr7j;$Oxlsh2*&R;aY5DU_QtNpipPewHQ3GtJ#xkOZ3&^K4h3Fu>hbt^UjYjtBd=kjB@8Uy<thof6w^E8fv11l$|I5FSljr'
    'v37Sr<5exlSok8%N){PEe!sJZwiBw94lxbR}S0alakInJ~`at#p+*iHM+ttJMCPSbLsGSV2@%e~h<+|VJQ$do%ZabC``w5@wT@irf('
    'Lz1za!Qt6-Wn($5Y@SOx1+tlk>Q88ll?%B!^PgiJ(P3iqesD$UStO}lf)u^<q{}%QvRN#N`JHu#8b&!Wt?$i9z_S8tr>0=i+11EN3b'
    'xX{%9MUyKbyR_yFx!5n@N0K|8?VpxAhC{QmD1(Y*JR{N{m!fnM-oZ=;!<@I`>Nj)ykhwD+8^G!1#^ytiVUc==oAIanU&@b+0J4J48|'
    '6_p)4+*IDvvg%XWkCqZ?_8xoVyQ%vv2K*AnA!_7@EBCBpsXF|*7b4!Gk$>_^`ny}|>!%$fU={-j2(4Av+IJ1ITd0X1LXhfg^=kH~t*'
    'WRZ`Ta(H8tJ2i%n*zP2EkTx5y{2i>SyuIT{OPSKIr_0^BJthVLnmDqq9i?+W^LO}46EEZxuSVgUQL4K-)Ep+s|Kl2_XCfc8qkOGpzG'
    'cXBd2}-aZ^8dNk5{m|DJ{BCoj-osxGi&;jrX$HV%J1g|Ch(faS@Xv~zYgJg}JwKFwQT>!&7MIDeEI@C|{Qb-helqYv2_J_((4RoPm{'
    'J~E+RguPgP67^OkLy%i4nivc4nP1<*#-b2Z@k^y`e#7Ky>3n|clyUI3Kb5{1=fXDZ?4*X*^BEp;$+UI>{%F7_606+`MxX2;G4?eLDP'
    'KW0`btwZlX9}#vl)oKKU4(1!I+zBWY~Q!rrdpoYnJuX&!#erNw^YTnmUSkS{cy#?HGKi`~b}xBk7EzZP>6of~&PYMxV#5hkAv0+P_T'
    '(^p1wJqW$g-20|0iY|;)i%W5QcMcYxgcp86OA{}0OO7l&lX7r~CpiHqBVT_}AR%(!2E}zWY+OmgEGT2VOD=5OOG#A{n;{)eVCJ6<G6'
    'l>Yl^ucWwbnhOA?|+7%s>q1mNw>fy@+&y@kpzG1hzO_!&&0AVOK`HrGuo7UiL)P<O6Gc8B|p}Su*udVjLnDrRJtqx8|Q_gkIoi2>7P'
    'Z#htu%7Sr@Z-hAUYbGm*D!Sx@Br_QU06U-8tsWO8j=J()Cr84fA+Bla0lsdd?mb#xMWDp*dl;~dG~d{dr(^$vd?6#@-EXZ$&DE%E>A'
    'jh$yRiP63y>^xLUx{aFXkpDP18YRGc)GmWB7CL0_s{Kr4umT8unE_k#>tXVNGjQBA9o8ojD$utAH^quWTjW<VLn;WudRyr4o=x1_22'
    'r>p5&G{vYM>@&I{fN6K<BNFLQ9XE)Z_d`dU*9Aswr#BFA6w;oZURadbbgQ%45u#eYy}ezl|!!9D=%cN6Djs97b3F421i+kwpdfSgQp'
    'GAzWu0bG^Na@t^F2>OxxR>S{{#!5b`xzjJkVjijzi9oB;wEjy<NIwMnIYMc=M4ROX$W;4^(xe<2`&%y`XaiWptj3d_9Xej;XkmoA7l'
    'O{>@w%Z(bi{Ey-$XA>R^i;>q;jeJrYDw6s=nkVxhX}VpgHD9&WH<YpxDG{8BUK5|@y}_HPMbyyyHmlW$P2H3%V$26y`y4*;wa~q#@#'
    'JcLYJwwMBVc=RVM|gckw>M6<DF4cLecyAp`nNTgj*;!4_5w-bDxFs(2-QqmYK*%jUt_bZ-oN7e{Y7y&|EWPe|gVO>pOr4W}ZaL!5&J'
    '$gbo4BsIzxIxoZ$BOQGTfg-piRf|e%N}<fTuiWW(r8M7D112lyb0v~vWYOt5(qx~&d0I6QF{w`c`Ee6x&9o39<4yJJPmD6t`&FQNK`'
    'j{z*+uVdI72td*f7e+A{bX0N0?uIm1XB>6Xxq>Dj78m^%_M;^YM)kw#F8wDV@Rboz)FC6AzK>;auqO6^7lGd(bgo9KMuKgB;h*5ZUW'
    'b;)-Xn+vdvRkB2|#3D2oyk+LIL3bItMyayc4zNffDoDVd~r)Gtx$+B(nxFy>W4DYYUY`rBAIerG6?VAL%rmqC^Jx=)Wk1mxl`)Zxv`'
    'I=V!a)s`_B@oW@xOGDaY}K{Igzpc~YxZL1%KHQ~9p?)w&dKPsdndY8>0@8_1!iEI8{524miAoVM%Pbn#C5e9Xr^U`XZH)Snv69Yzs&'
    '%WAK5WAb?=y&<HNxqzKn?qvVh9xm3ZS^J7E?xm=d7~<MvGiU27A_@mqn4TmW|e-9@V7Q<%R;_Cd&F1-yLa8oga~g?w#)O9b<$!{}{I'
    'GFyj1r$$Tmo9bD-T#!y1jl>}*D~Jvmu7TG3C0PF>nu?D`;cM58)aX|b{<LyJuKfav4yEI)gBu#WpJhX^em%E%j}*5cmf-5&X?QW-fJ'
    '_u9qB(}<U^23suKy4KizSjlSiG5v>3krEk_ay9o=qm!ib8@*Jj52i!Z)sVU^9!M%_qc(-tv6(311AEZ`-+%JXH*|G2rg3RN~8Wc5vu'
    'liEmpS;JHsYKGQwG+|ACy>QjrU&DkrYY0*kl<2CrYm8W2{@OgN?w;yj@lAy!)t6=ZLd8q%j6iafVp{yHt#qI^jka4_PbuJozzk@};6'
    '{t$HJr|P`jPVI$pzzus2gB;=A^pb~7%2lMxu3ZH-`phY{Xhw$7}zEq$3<?7f+?o?FfeqP6zQd+m`E^KpY<b?UtF{{E5Aom#?9evEz-'
    '#Ry#?4_w-r>HH$lB!9+kOb0*7)9p--WeT6Uj@N#fJ7+3pieowgIwGqX{@u!&6mB2Im0W@6Ny_h?8cwHZ2w(!yUrNaz;r3$BG}aU3MO'
    'U&HEiQ{bVK17!U<iD3^6U@37RD||oWmmPW-S$Gf>Vl5!P>I)gX>Bl|L(uXG&8CWEB7>5rUBi;)K-?xg`yFryE{%1yfo3B9P2}csp9Z'
    'v1uBvDuM8|1X3Gh~S?L$OaJwz?!h!!~;i@OeT6_h<r-*7Qb1KZe9M)K@>chto%=qKRxa=k~T9Z8&>)7}3YP`1FDpWcky?(L3B!rDOQ'
    'E^S$-zL^0GWxDL<1#KLYVYgo`BgYL=J*!yY?jhc1{vpS;DX@@C{9@-9?rK{+LdUL$G<}M5hW|6Clc%nszLDZz2vz~aByy}mjGV`yXh'
    'p87?+y4&D>tEuOFUKeh#KNI&`fpcooZG2#E^AFX8oe+<mx?C(cFRGGvt5VJtxL(qvzuUB<}7x<emsA|-IoqlwQ&{Qh2-PdFGlVg%a^'
    'W_hnI=RD4CW`Cp^7CBaR%VT`LXPV@(VA3U58w`b!jcEPaLktDBfq+1FfZL<lMQd=G{>AqamYgHOjiu&!$kZc!j`S=)pxd^G|J^U|Pe'
    'O9C6No?*Q?K%VuNO(9F9N<sVldJNzvvO^u0V9y%^EVvy>B&**Oy;B0b&5hq6b0Zw}X6*-)Bk4HZuEu)2oi!_Z>j>R+uaa*2afH6A_5'
    '~w3AF@EE7@jYX;C(eWFze2#W3A^pR1}y>@o+X8YuVt|ANpK;r9SUDy%qMI62JgkG4`c7#b?YSs-hx@W$IIKx26qAzoE!()0@mb=$g&'
    'eu6@q67z{KV*YE&WFo*U@B5YES41QVCN@Cq=sNS(Q43{>=fGv*DT9Szelw{EO(-0H+;~AKz6vMRuMPl^67XND;PnK7z@K^JKiKAyNc'
    '|l*(%%)@Pq<3NX{X!)(uVgX&kUu~><6O~X^a%=v8?ud6k7#)CiH7R>dYsTdofW$kg`CNM_(XRv9z8Pw=auJR;?1q_`0q^|bD7F6Neh'
    'BM%2Pn&*8sPDV-GXGxe>CynX~eT_26yB7OZVd$36OU`I>`kVTAxkLfV5!%Di(ZLi51%+9OW0uLAsyOF-VX67N_Tfo<A53~JwnVkw#M'
    'IQk7a^7IOE%9Uenmv?fenJJuGQ32<Zc$a*<u!qFe?Su3E$>aw{;LH#iSif}&j;zceb7D@?zZ;_P?A<n4lQj=VQz!6p23pX+#s%L!HG'
    'zeB`|xmN0*RE7;j1DOC{YmPe^{G?nbJn=o%5TFyez4g5Gf@AXS~?s;a#9nFAvp6uaavn%cyAjOH%0mf^3<06>J9A;kVE!{Hg9P&Ob<'
    '%)v;5BZEr?Eq_YbZE)~!kK{r}<Fb}Hpb70BrHZrp(os3(W!5cqW$@?y@hF^P%N!bLZ{&P?k?41*b2S;M*4CQbP{F8yb=R4ul@Ivr<q'
    '{fP91FyX+m}<U!3w663aa(IFxo6)*q;oHmEyvSP=*tPZQ%;<Hd(DXb&&h{$v`iv9cdq5tJ-xtBTnyHpwqk{R49WPQNAOThpSRl8LO*'
    '=b;a>=wVsKOuFVJJh27X`3|H-fBuO4cL;2Sg8Uf;!t_FvJ~PzH<bw)0QwM)}QKr1{f<n%v9A40vh3jk+B<%ztW+2TOy0JMwuBf8z6B'
    'l*lJkd8rqC+%bbs7x96Zml?Qh&SpM;#T34{!;hc4^gTZ3_p*LlYpK+R&3N(kAPfn1qPS`+@3#ImE!>>JgeuAM%+32`P(FcI{4NYz+x'
    'x(*;2e|~J5ZJTJ}^P0fbZ*4V~uW#v8x6TlX<rc_@b-D{I&!R^3>J%)S`UY7J;DIaEbX*HVZeIt1?H@PjPo5Zo_Kl5o|kgoV}H^fVPH'
    '=;|sQ5AW}P`aiDHK&XL^3CzpKY)_6AK46P3E?9SsH1g^7RjlV$ea$DY6KpfRcCX^&h!42mogZ;%)oS7fR-`ubTgkRRcwMFSLG-QX~W'
    'j`7oZT5jUQE_-8_s<FJl~}o>8fZG+3A^{iz=WjvWc<^Y)?nQZH!N0?ac=o&IS};ELx+fZfiPwrI1QS9hvD{jF^JYOM=j^^@cg(OyGt'
    'sHPMJ``r7iJ=!PqcJ9XkkzU&b;DapOpCMH=k!pGv|j|Ds`AAWe3vWiqGy<)VJt(Yqq%lsBk?eO|ug>gp62wbVk@8B^(wmwKGzcSWAD'
    'spC#^O%U^NfSDm+1m@B2$c;P=vQkt4JeU5ZF^`VGOoKukFFnOt+jEen&zcMUHTy8%r=Q3@-GP&x?^9csauPeO4cho>NI3MD2z&gbxj'
    'T1~*Z^N<wf_;=@F9ZSNzcGtclNU>jWHN$p@R*f>SRG<8ohSXfXIE4L;ZO^P;>D+o%QeE_`Q5WGZtjPly5e8c!4ONE3Lw*6$F5vQ!R~'
    ';NhN_dUGVsrFZnoC25Dak?wFW_bCy!LcvBU{Q<~wG&LEzZ5MqadXTjG|ZT5PgDsjseLyOM)V5Odl{qmP_uq_l!f_uSnLojZe_msQ|e'
    '1PZgmD4qQWZCJ5IY!9I8hY-8p^5%<JT4p0uJtv91RsVF*F^fbGMrt#e-AigKD?Y|&hJ>e4Fk40@!n#S;mC{(Qg%?4*X>-6hc(Rj<7_'
    'F)v^~P@$)YHwH;L8V+lW)oZXl0kR1jJ@>b<iKPgQ&Yr>)Xpp)Jn}nTkMlX#s6l{=~Q*31f;M%YmxFCH$n+NY)L<p>~TAuTge`F<-?I'
    'i#I-GN7NJU`lT!)GP)T&&i;lis^;~U<8}BqUK>fxBxQ_!r;dFK|HU+Jm?l<wp--nRoc0?d>Ee?7fK4!66nhHi3hMB+LhE7w%CB_kXc'
    '39-+D`q}i?XNkqWHNBITW3jgJB|dP<4JcUKW*Pb;aXxC|naWH}-?u3Q@K&&=kj(XV6(c)M)4QpWM3jVchQF;(xvsPC{B#=vFCj+WuU'
    '}y42tv_u}YI@~qT?antIvHVzwXkg|y;Pn-@gqu(j#`S&bsH{VGntMqWUBg$w<qy-KBG@1CXpG!B~ccf!)#*z0UJIQF2KWDQSY1ZadG'
    '{5{Wr*fo@Q%oJCG5YI>ZgviH=~p+6Us%Dsn<Y-yABrOOO20YjQGIGkMKGx3B<(y?Y57IQfP9#`i*}Y=<gE8zr0))$p<$h})XJcYUbZ'
    '<+Uy*S7qI4528#ZPR#HBIsFHWP?@?o@mFq%qX2lWizO2!><B~2<#ba~7NT7H01yG0&!nWr#!rS=lNtnW!SdOqh|(sJn35uWy(8)8bY'
    'UMHV2!@0PsDyq1mkTGlUBQ70!^mK6`QD$1`=Wl}a(wn73EVPPlKk<dLxobmL+bpJOF8XA-;u6xef0#D<*3fA~hBWY(GM&<1LQQP$P@'
    '@U&ndFzbILW*e&a4?E6UYf%_Vzd=)#&5b{X8zF!_4G}-F#o`L_8)W!f8)lX`TDZ3ym%JVc(G(pf2M{mv&{Lb)6PYRTJc6tP98+BXc}'
    'xT1Nsd6{BmrCgZldfOM@ua_na>(Xc8+`&1z;j};<%_a2i=&yA7PKcX#?nN<H)1WHOaF*mi$K+LU-*-#e)`s*d>t}R;l?^b9#cMXgNR'
    'yWME`$*64eGEU<7Eu2qHh6B2EN=BXgL$xy$nBp@-6~9Req03P-u%XW+_C}eekIe7%dH{B6j0S>1+=P1f@F6LSh^ghadR2`*|!3pO*}'
    '_$---vTZMG2Rbg04Y^+%%k@jPQ5wuJb8@q{v!Kd|m&JQ$~J0GaUZbX(+6T9<7@`+VXU$DbA$UhsxY>f1|2UL*NEs)d=Y?(CH1gXDzH'
    'FYd{2Q!-XL7u26`;Iu+LLGXJHIVFCctGEzMFLZFw8te%h&GqqS@pA5u+c|D-#zH1uKZyJ|YR~4jUM92Tty#U|P4rcg9URPANzHe4LP'
    '5GA(H^zn-D~{mpW;WHc(EMD?Ms1q-$YR;G8ZmH4w0R&rC@ns4P3w4X}v0DCAoG)9$ig8Q|(l3j99e@S1NWAk=fE{drKI$bPIt~<{>m'
    'aDhVQC3G~R~G$L)h1~+CbCXb7|$iuIy@Mi5d=2-V}CZp&m*^go9Ydag*9o`^UDURN<1Oy%!(JW?=(-<;DajuJbedi{OACtzfb*JFYj'
    'yH7Gwl9og#d|uXzlBSEmxTY77II?Mn%s#?HXvdbO5z%->6eO7YY*>bT=t=lWShu&dgpSNmC=wCbscw{J-9}Q?ogG*(BB!{$6cF<fRh'
    'Ee>G+Iz1plIf`O~oEU*3bad#uBB<Ux2lqR(1mj8+q(S*4#y+N9}F@+pe6I0)jwm(6eqL+Px?*P%gYDq09xfN^><WbNi)Q)V-#=N5|z'
    '2M>T@>L%P3ECBVnW;`4A9b1q4;qCp2WP{f#x?Z~hbAzN=*J%fc>&1!S>9(2Y4&SE%R+HJihQfT8Ko~W@vH-^qIDnRq58r0`*E;oE0y'
    '&v99$(hf;7`qu<W_SF9eN##%ZV9rRqP`LedE}ilveB*?&E$ZtizZF6KdiRhpKIp(RmX^tsC!X!tOK<rmOQ}A{FH3=Ib=cyq!)zu%6U'
    'f`tj$djK?2|i<w$Sq}g|LVeqyiD4U%FR%sG1^t71%XDZ3d70)A?A#361gsJ3QX*vx2jmOVF<k_X4i!t#`0PIT%C5lG2)HLl1^<1-y%'
    'krur-sjFYlv+jNl3mAeqUi)IJCi|@h5Nau*Ag0r=A~oyA`5t{n?>xd1>opEzFPVeaZfGJ-l>=j4^kF`|CtJi(pJV*X9e+Rqa95?w41'
    '#2nu(cypNPt`eDbVDh`sOQLc_usrsTK>`C+2S2iR!g%hc;QDP;n0&``ygB}`>b{fNR(seR0pH<!qa)0Ggn)*OfQD)GL<ZPKn-ir&@k'
    'SlF76Z}+dm&%frNVq_vL_Dsg+OUq!LuO$sCpN=O5gV}>E%Rtzu2*dyR`t8JTL^w2?$%&6bE!!gc>2@@Y&yR)bh#1UoA$(?52BfWcL!'
    'Aztfp>@V=*5lplr=ZTh}j;fx#9yAS-Km7()Cei6~fyxM{Lv%rSY3HK-*&#T0~3(A^%ilCfh>SDQ!6Q)f4oiPl033V)Afo7q%5e(}#t'
    '{P_Q<KT+^z8711}T%D#I{{!=H*?%9GMp4CZG2fFCYJ|`x_b02?+jmOT)UDjWlHe-#|1$y|9BS`ujK>wsDRF7`NNh=*l>3+(s9G^?x9'
    '?1f?GrLi(T^sLDV3@vLar9iQ9#o$(23a>lEcs6pOm}X<$?cWkVQ>N?<wR(ivn|OG`pRu{3B%~A?}$O)I#`g>OiS(Zu<G4hMxgB-k$Q'
    'iI-a4lM2Q$y%l73N^&U^%H*dwaa)DB*|t*t$NDL~71892e{q1CcmRC=Hu!?&sNlZ;L<ijNn=HO~YLH{HWE>VJT8^>rlMT$xYE`b83_'
    'IbrbC3GAemy<FPQR(eSHEUJz65V1#>(dTO_Q)7IBBp%)aekY1Zq}>i~;&&HFSSkb(lco6tfhck}vw-gvwqX*T$C%YW^mrfe1zSa7c%'
    '4_kDSW;`PHHt$$2W!a4!fR?yy#%O9vDO5KEQdQ1QVP^V7y`lruUtPs_7!2;B}VB3r^xUO5U^n8(~B1-YSCK^D(Z)sRm3v?9u;G8!|I'
    'DkujGf&gyX}4O;pTf5bOHxws<p#q$>g_D|qfIhV5vDUA?gxtk0tsMGo)RlK#n6c1iFia$K+(Q8*Vt~q%bjg{4KPoo3e>gWJtyC0yVt'
    'H9koxfD$ftCD67LtaxO1-mNpvGJD?^oTj(=GJ&lebENu>++kbFp^}$iOIO}+&t(nze+Fars0go6_jsw;m&40W)|&kX9|RE2+q@iCDD'
    'Hy`e#|f17~|k`u3e@T|Y%X$#V3mn-d9e`OA#0ufz%WGU)cTM)=gE3WDBB;OJj<ew~Iaok%sX>DMI4yEX^qM$b{fqA@zOeh+irTL{hn'
    'RKd`6Elg;<MAAD>HF(Cw)4}%zFj9XBOy+EYoEK7Hb^ndE;O2OEAXUQMzFr81q3y(`q=O3XO+cai@cP3}%RznF67WxPCtk$|Ail8Q`o'
    'tPhxTEMpZ7*M<F;Q>H)_OO-#rqB|&!mj>$jf@`kjGqL$!{Y5y8!<bIs)F?KpSTXLsd%?ow2u<t684P%52mp%nn<st?0s8%Ce;6Knga'
    'wE(cDwmpK*j((1YWL%K^L2j-|>r=4?BscFg_SblPta0{a0%eZTh6#oLK*k49)PZBJoE5JCJCnr-QiFjiLe7NWX_PW36vWJm)=iwqq6'
    'BLBqnw!``#X#_RF`rDGb(l`b_k?XbN6DIlx+KFhf(m+C)Gt3=j*Du-nJ2XdTuw(Pv2$~TI}Ve<JXej3{0TzUcO_OgmTaXmuZx&-K3{'
    '27k|{KfEQ1|G#msCEFZ%060bQ~}k1Qz2!7NKNP%V8xbb6n{eAO%e&iz4OeXF3=&Qb7WR17|D@q-|{y|`;814U#z)o(#E;d>z``t1{K'
    'w~WM+3IVMB;s$z6Ux|-uF1?@EfcoAW@$$P4uDvk=jx}2mW&>^bWq%Cs8K=X7(KGmTVhN3C=ZNxw`P5z0g(x+JaaplJ@afAn8e1@AwN'
    'b+g`~?=#WFs9uecuOU8`Q|O8EW9At-yVo^poz;&Zd6%yy;uVLrm+G0M2n>BByCIp3VNPfa^YlA*W~rhW1fV&~HyGuYZNF@{>rcUJrP'
    'SK7@TWa$puJ!f-BUNTl@?ICFhEpTB>M9=^Glti0aC{0cCDdw&^h>WYPD0^_N`wX0OYHv-Nt)J2PRcbFFAjW{YhgJoQ8F-H3<UHn0h`'
    'hAL|AIw_F<qe^rB76zXKS&_1c@t5+Ydp2~DZ<q&BEaNZ51BeJOe+5SLOe^)(hkv?*gv)&4WwM)ltu|WY|3bG%qaj#=~|+Y6W-tv*o)'
    'n3j$%`CJv<aFB5|tLF!lYBhH(Kr@o0(1!mUO4O(>uECFjw*dEvOfTAG#wE<m-Z8Pt8lH7fd<rBCkt1DKV!LF_|4t=Msn*40<Be7Xsw'
    'wilxW1LVAhGKn7iPJUG<qjpLm_%8>hH)k5Sd|JVuELsL~)8_Dh-bjG4$t40cdvU3N9_h`RL2Y??9L}4<`j}^N3zys`lC}TgY2kNd<&'
    '^i<O7`Mll{~;WKRL(`Uw5RAcP!D)G>Oa{w;7*q3L<`<5{%8=9_#1T)3D<9IVd{N16w?F_@i}hP?WS2>mSGANceefT$vrSbgCx3{Z<Q'
    '{H_Zp#j>lx2!x`)CMxU_EK@qk5f0An<gQQ5Ih537RF*w>=V&-rSb*ke?zrlVc`t1%9q_F@OToB?N4_u;0REluh#sFxLXyXowvt-A@o'
    '1BY6GpW~!$L(>N=<GIu+IjhaO;eS%VC)QfF<~zGcs7$}N`GgBh2>#Ns|78wOCbu}C$i<mPjd2FEH__!1O_x0ka1p(G_=(d-&D__m$o'
    'S25;1MKKE8|7WM@GjDq_y8YEnP^o+ut{<YP6H7&q@>>e|*w-3^cyRqh}OfwHhVK^%6?F{R%$4iU@M7r?^%6{*E|_IrZ|TIflj($cM%'
    'b{uKp$yCDXpQnSvvGCuIdfe7>jx<%rL)n74*m1g=3g7iYB6XDbeq&h)<x+4Gc4o(4F(l$$7N9Lq2D(*KKxnQW=1lJ=2DptziP-bbhP'
    'Q~#fFKInIgowg_N>;A=h%=o2)fyO>9~JAl2#ze%6i@8*41cH9k*Wm)ohG?yFODU<SgX;+5#yyZZs@m4(Rs2=VpEWip|eb>+{VzG4;h'
    'v_@;jfWi5(e%WZ&1>&~F#*^AcEvb|*bWOL{h)50?cMA+kgli=;?@vQ&cMNIX#OsJ5FAXOi~(8XJ-z`3c0Y`lMy9*Mk6BK50qXWU`>B'
    'Ktj#&%H&J?1lL0Y7!uD{2P^bNTBxRN4W<z%|!9e8eDMvIiq-T2Nq=7k*bx?xF6GPxU=V)*p4Iu7eSmC+F=g*7njqZ2j%F#&wzAoYJr'
    'tm?*CX(3`mIr@3?k7p3V#7t8Z1%qe}Cc%0+d=$Y%yKYj`g?sI?detL_5HSArWC^N3Mn4c406$CY5j{Mf06eDPM89~*#s>;ggVUmu+H'
    '_GM}oJp`qJ!!Tj^BzUUUlU!i2Y^N7&Oy|iIjbC`eP?@>c6+?>Ch1tQ=4`4>30mv_M#Wa_h@cW1(e6@3-H@fQS)yKnle|Z|(>!`!yke'
    'PhJ@AsEy^hm*vA1sJ)<H1bu3AN05L`3$B;r#<kxcBOnRBodTG;a)Sc$DXbv8#X3XAuXu=<(mk=hgF3Zs{73|FZ^KdOa+Ct=5tWGEGG'
    'Aau!Wfp3!iATp3Z_FGzHSEorO=wN{S(N?II$laF#4c(kVuF9qqL<b8EiU3Z^GZPx*b|AgVseiL*&cE6rX7~@`tbXh%>zD2)Gm!TI{C'
    'DAu6HLTaiquAn?hhF<e=!NMi-0df)=!K__)MnK+j#qffHE-*px28QOKL+&K%#|7Zz6>ee;mc;~+7twvpWmlR5jU_WdKn!Tdk}WUq@j'
    'c7EO_+Q0Ow0i#H^K3jLgyz;IGZVzp1Ygv;_Is3$2$Q`z4TxTjN0g+E?;IvKdGF4uV|z2NZsZ;MyCGO$W#E-&4F`l3*U+_NopK+@UZr'
    '^D&A3PXa3EjAuW~4&iF|D>!`nEFHTlg&&2U;f3L~;O23f)_*BL!TIm0QtD$|urmbO%sT1*(A|8>sY4L%Yl%UZT8ZA}B4+I2OddXqf|'
    ';8hzxnB9(qJ<W%lj|U_Gov0|Ncy{)B8?J+@29r&KbX1uA_DqE^MIBJ&;<zhd8cn;?A6XN{ZzriR_32-2e5Ln1(u|o$?;O%es-8=)Q&'
    'Am9xpt&4X}90eFVXL`~Hc>U2hu)%{xlXa2g-%IC6>Ex3(OF{|f2e0GC%P97W{y-X$t1ktU-rMTLxmq^}9!>buC{IrHu?4+efNx?4x)'
    '_2Zk-gfJLtb2bNR+MbOqjqsz^t$6*xxXy&edo$=IC=%s7u|=;p}HV7b1j^!P#}t~qI|w3;b*=tN1^V|kRvJ$DHFB8Gf0A_MLofR%Xh'
    'i5a7F&5MIP3i7G^g{NutGGZ9b!O2059{6NiU2^tD<tRg$$KcHNHriut#2LxeLeRxIY`B?*9sxCb@!KY->ZYdNXeV$ipy0asl)j=NHC'
    'bKf3K;s<FohR!<*%alcM_eV{bR5}5ipFhISnpwDC{~2f`dXo&9DbVcy6fTZ7pxaA3oGEq;@;wft*WWc*Qc_0Fx`^?a!!zIo*$fFQD&'
    'eZ~8?29rfL*y4(K+x295;%Ds?2rpSi&2h=-tB&yM@uy;T{_9ngAIei{MJ9Gz}R{<(BE6CY9s$NS#PE1TIn}*XO#huLhMtWKTY7zfoY'
    'EvukMU++<qoBSIZ?XT#-5QMls$elU}iM}@nTg~w;1VcY{8Ty=)*@BT{;YIKp=?cOjGYGC1*4h9<8kr`pzpgTc`oh)sD&8^N{T*_Yfy'
    '^n*_`O&!9ynzk5e**1%3Hg+8nhYN)1fvD7$m`0t<l(yp*!wb<=TikiI<XY^w0w|*TDa1C98;~OIdzwI`YrkfSvd9z{={aGoxVrt*Yg'
    '$Vq8kRs%G6+)-BR9C@C*G^qXSPwCquw8ciKL9oyJVxO!(zIR`~zq0B6HyiG`puG8<-vc>ub4-~*vf=2gpk@<(F|KWlU*JS}|<uHDTL'
    '@vt1c@5};I=}b<QzJw8_@u;hQiGDucgCUvOP*PKmzx0a0YEYgxJr#>H=59i7Z9_P&tpx8JByn5kc38Ik4?5c}K{Z2F^26d0l^2Pir_'
    'TOn{9Nu}d{id;<HTlsvUeAz)YOoL<rOsKv@EXJXatpN191Fe5IR>Cl9x1zXa4B2KgSL5`U^F9-2;bVvxPd;u8+j<O-Jad^c2)odjS2'
    'JcA(R*(@>!N4oB<;VMe7gYA^asO0B}^$lV^=Kg}Ok^!<SCfwOV7@H<@HB}v*(crj)nzVz9niLB=I@3bMK3Tq`2Ay_4dKHK7lLaoZ|&'
    'ljp7yRo?8!?-ZIeaZ^xGYlk|UC&82Kg^^o`AK%&^Tuqpjovs_#U*r@kfV$YP?g)n+3Ppr>_YTBvdy}JCSqHg1I_3dWf}!W$!{fPkn~'
    'OlnV^?+>_RIU$(jl;Opf8{N>j3VTPax(od&K)QppQrahm$4m<r1&lg>IRI;;3E=Pg}<!Ho~7{x=~^un&WQ>+`XAYXVvN(i`oy3`ml&'
    '3b|PvMsA8l(fNH6=(ocgePq&U4Rs;GC;~QN7r3Wqf-tha4I=k0z|ozeWK`@F{P(+@CO)`G9d6k$CiA1gN~D#lXfA_yJFdgjX}WmM=o'
    'lWZK8W2H1mTL$B>Iv5VJ6?WLtg2v!CB=AWb{fkam=)ZeQ<{^JGd10{JlklW>M>LnhoSKe;+hwPlhT7aW-UO2u!bTqT>1-bN@s&toF#'
    'ELxLLcJ)wgr?fZ<IPuoyY`8{M{Sr`#~K8>vqyl+jbo<gi*95Z?OeJ;vj9&6k7h*Mj6oE+O1fcqD=6RU1fZt9IAP%5ZG{Whg>$xq#>N'
    '@6=C=`NuWhhpip{bAfu|3M;maybNfe<JP{%RzqaVz67&N>^AFGrzvdV)vX2IQZ5A{0u+92k%DcmR$tSU9al1&)?#tt;ES|(JHWZ-$J'
    'fDe@Xk^MZqqQBC2;Zry=v_eKen(h);$b|IMc^yql*^Pc~ZPCfiwjaLN{Rr>C(cSd2{i-A(UJd_f$0b;)O+)3|E$e)gU1Pr5hpFXbO5'
    '&_l0Q5$9{V_-H7dyvX~Hn8+tXhRy&HEud65EEWFjI>AfQQan(U%N>7ojf$65UpfDUGV-RT82q~!mWywvy7uvKetj(!I--EZLy|Om`+'
    '4ri?n88MsW$7g@FgsAuLp^4H(077jI%}^p#9=oXg`rbQf?}<e&uR##4Z9~rR!q-swB`|ZHUt9y6nR3N+9!Bi)eq_hUj4l=J&V;`Jdx'
    '(pMW8~XloCVcK-N0LI{6`FNWx@a_re?gXe?S&`AC{DR|+CruPou$+J;JYO)i5ea9iZducqi%?Jf^Wiz^{=mK4^^<Oqia)}UmjX$mxf'
    '=$tDQmetj#TDPNt)UUm=^D_Rn`4>n`a|4LPlTRHIT$b80G91rtrgcC!=KSHj9y*_<haIRjp;<*DO!qXDolZkA(e1${V`}+H4*hLY_z'
    'I2`@(rtub{b^rbMaPllm!1vd!UtaFt{UV?N0P0-TPKn)M8bYfnJ0+*+_qevB?RZZaN=CSgq79WukwfbVu(NHxX7xZ$JOaHUNan%buD'
    'Pfkgru6HD!&Av`e*<LK~Ou*hCalWxd2Bam;;FwA`AwdhsQO9q{Y8*yqts-)1`dOT&A`7xXqP(r&7cyzm27cB!canTzE*|)%%>Q*c49'
    '1Jc@gDIFsE{TH(Z}kEzN0Ou<t0Mot#Pn+EQz|Qg_A<RL~NS5ioWg?#HfcNRA){rb7WICEN?wbeQq9roEjnCG9(LcH7_GVN31YXkr3S'
    '>Td?p+!pDM*bkK4XtxKZGr5yvLNn%>Vya!S^+I)#Pt-eEoy=u6&<^!-HE{kbCJqB+M+<`e>p(wbmo2ohM5G5la_%J^Q>b}V{txsNH@'
    '{~!uo?kfr_m53F0tiO93cygo5?;?@l&E^GqYG3;*_?tCBwRAlI@<Ld<oU_-V}FgY>75e^Yr8@d7PfJM!>hR5wk*8d*Fp3Ll5mOnIKD'
    'UNF}$wLAO>G<g4;ML{$i~=eN=()^2P*OF0~LlPMCn=vnTlT$69{;@+Z`IMGOe_h*O34uPB9KT2>v7d}attix^Lw*5@#Nb8ZvvKW%i`'
    '#gVR8IiPyp1ACpT;nK}5xaPY)e?oZ?Mr+0sE$bfKQ?AAi?>kC_miNJXiw<mU-b@~6tt89sCsHPBCON(E5m|dQ7G`{EqrO3xL65(O!<'
    'ilQ&X^`9EpTG)%s!5nTp}>}aRb#Gmgl=4NP^U+7jSZ}3l}0e8O0i3LxHg_@jm&DEWNcGP2S`|<lzaBWAc_1$0WeyY(Mml#|F_A=csw'
    'ULxbYATd<^KA?6;NM?Z!u(;Y1~7$O%-3YI-1k>{piedbmyTcZgN%w%z5=^e24Jq-edCAfC#1v;j)0|V|PL8ovx^Zf2w-0bMgZW1ZOg'
    ';}DwI`BUEg9-Gy{(Sh_DTE=OGugY>7vXS>9ynezrVqj<vR`6%^FudoGcW%hq)*ND87(RdYH}6y>3li7#)RO(z7bsTVIME0U5BoLo_I'
    '2@mP(fd!Olf7K%~FHBg;MrHN3(O<cDF<^G5i-+=_Ke(1TZn@_bKO8*`#Dhwhm228$N&L(ad7%zLsFUJqEa8#~6V$E3f*%H*%86*Glj'
    'yLAbxCwQ6}O{%AFUYx|`dF6OB<^Zki$bitMJR+%9*|3h=fGxdu@ynA2Ouwgq`y)N5=sYXV>g;>`qn!sf6>Eu~`+b<QU<pcniKB+WhP'
    'bN#40<ppp!=XETm8V3&HJX#j*R!mTW0Q5Do+a!M%l7LQ&S;cT8lpuXM~=IPNUoBwU`%DgW<l+KZYqpuLIp!6L1Aef9S%BJ-2Z%_6){'
    'FyhrVgiA1|s5$7G-1j)ZAb0(R)xxTj*tj9<^jF=ao-i)ufYeEiJ|M3GpRaD~}|5U;3zrWG1xsT-T<lwd4ZnD5GnqU5U9nS4vijO~x$'
    '4Nm0GzL8A;yur)dY}z^-$4VkL74shkLR@OlX2-h8yGiW#TFLDk?v!ebXu+y%6j+WOu+_H`@j*bJ~orqz+4=bOM$V8*0?=v1ZLJ}z^~'
    'L-xKSYq{z|lAkiRHSSzZB`#I+z-b|nb!9M5WvZNT=a>v7XlB#w?Fbl9xN`iRyZ$ZuKAI9z#vxowoNvof)F{|gekvz!bZ8bMp<IM7(c'
    'F_ju3{H!_~`rWt+_3yv5J`->k)q^r=e|;HFm#pIaql)m`o7<4-zrZ@*r@djRfE@NJsWDymmar!Kx*$v?lFnT=#3c_afLCQI`6Ba-Ie'
    '9$`<g@<JY-J~2Xqg(;cAK$fN7jOcSr0gDRpsk02h%xP&DPVNHABewoe)}@ObUnc$-dule6Vr?-w>7vxw8Ucv+i`75WWC}E3V?4ISHW'
    'mI27Mk@OW^E7#q<ZN&Yzeg=8m|*G{p(%Yv;Cesu-re3#%NB!!@7${FrV@l?L>b1wGmkYG2pIP;Y{y5Qw*0mDs6*eN~^$%YQBdYpved'
    'jRfU65=glH{u8POL(Hroa-41;4ynL^d(JZha{q?WKI&*Jb8>x$#>wRTOHt#FarnYXV5T*bEuPKz)LmU=7M`wiL&#4d}SjEKeOX8(4v'
    'VXg=EvL^Akv5&0g}jrx1Yo47LixRHNxN9ooDGmLItVUnCa61NIzQ9W0Llv!zf-G=si!UxurW7?3`<aUdTd1nE&xywkNqe8fRv_*@=G'
    'CafhmW+Mf16RYq@t_z60T*0@!SHe3-r-99+MYzG&7(cr4@G$&39J&VRb1xo?+NPpg>>dcbyTf{CY63Me{)I(7934#EK!;0<ININakK'
    'h;BcP7w*aU33r52dk*p0vuu8DeAH@vUtpEPLbw`I7?SVcc~zw!X#=Nv{Kswbr<Cswg=*XO!-~8jex&vZP~b26+2z1yf;v>ic3BPLHV'
    'pkFS4eUf?jU={85-KSi+byc%&?-3&iJx}y8-n@n=uMGR_HhWUq2Vaw48Ff~FB%bXX&WQDs>WB-~y+pq@S3iiO5bq!|KrPFzCVR-Cn6'
    '81~qh9z~?+~MS8+AFL_i!VF@As=u2Hg7T<+9wS14;`?1?kt$-Gf3&UA@bXml4{9YxL9gFac)S2yYB+%qxK9qxbJj*{=*o0YyK6wXY4'
    'MiTKbOEOl*OpmD;FOdIcI*h12AZTR`qhAOTwwK3K3CjpBB}{!LG)n_(PL=#Imp@G_e5vJA%>dco}CDL9ayKuUz4;j$a`Oz^gaXvX}4'
    'XD@Xz?(Hllc%&EmMl<QkFe#!|CPxSFrmz<Kf5FvzD!8j#7(eJSWZUc*ZvM1f66rGt0{tJzuIsVjZeYkiD2#w<{BE9lbb#@=lf=8L4z'
    'ttdk)$gU_(-*b{>jaNmg6-ry<VBh_TEDMtJ|Qt;u)rAp2Vv=`Wd_X*Euo23$S!Y0150kg6@Jwuv2&xf+BjzUb_o;tuqB0?#+PhKi&D'
    '`-}A`2XiE@HUBu{(cEg+XBK*p~a%`E+0L}lHK*~?9Wf|vdbmf`Lv^R^#N_GJ36LdjDU77u3yq21~#n4zSX)^Yp1|_Z>L&=J>v|4ruK'
    'iNf4J=auTe61OTHfX@@tb7pOFcZ(VMbcr5#)iv+)3K@g6?r(61`k(6fW!j>ZoB^@5-DCwl{i`4U$+JVZzJ=`cN0FY%>#D#I$YIi54X'
    'biqt|B(NVKZPHP`3j-QYc3VN5!Ttx*G|ed=(>{tSj(3?Leg$I)>A1FpzYjh3oNks~cNY^e7VIQv?X8LxPUPK_zC`YO5*Lt@hC>y+{I'
    '+_9hZ?|wTd$cW%A78$T=doS>%w$ZqF-&1<T=PG&2IurY`4fGPR!+y21plYxk%mc+SVc~K5acCJ<eT;^eVm7eNuZ50SS94C+*Fa5OHz'
    't+XqwlGcIG^7_^EV~dEBtfCjRQ(p`QH%rR}o^hRE^2nFSn@Zfy1<=@HuraD(7VKS{a6;<mI6Z{2)|M-FyBpQ>LmhVA@PZ3@_pY!7Tb'
    'G?kTzb+=>Z%e-P_NmGF@HnTE~3ElFxCM@8ZZEbI6{UB`RU`5mexUXxPy%}<#3z0Zh8s4%@%W`#cu|2gRG-(2kaK{B>f2tOIvpzB>pI'
    'Qd^U*I<*|@aS72)M_6l2OEkYWWxZB3(SJQ1#|F0SqXI9N&y$$CB*B=d{p!uY$($D#5_<xM6?!q!E43i@O)P(^SkCdb9I*%PDyR%ZVL'
    'W$^w-zuwe5WbrwhS|#SiXX$a%7Tb{Y{1m`D~(M9wW-k+?s;$%w2g0N01l$d=*rtpC@1vNdTjllto@jha!+ysePJr;`%l!9+twcd9&N'
    '(QUwp{22V_VgjtXt}IG~%EJsM8@&3{KvYVG->`c^quU|GfsSa5+wvSmyh><!wKCIu;RU(6s*o(ZX%DVqY2a&fl%%zkTKBvX<`ulBLF'
    '4o+YCQRowcy?*_|B;f)#gON$U6a2uUZNvx)FScNF7yhH^b>#m!Pn18oM^X7{CAGXnD*xh@UkLtR1hD^hi5=TIRtU{k(;OW(CwfZ5My'
    '9Vh@=|^l6{TJ@Ps|iI`rq;%^qp@YN@MamgfO6luFluI1)3ec88Q-DfS{RBjKtzs|#Z>WQd&u#vn{93ZP3OQ7FA0*BwlAX%G%&uaF<c'
    'W-mZ(%Xt#PVjgy-5(x_xzM^7X&_WL2~@=+h}-Wsc&^^Fae`k3UP%#SLmq{2Dsrt{yU==4*7Xn-KL3Po8aMIIb{#yPmI3`w&!N-w09G'
    'ZTpCcEQVctDqY_IZ%u9h?4nj6d9!d2L^>m(ID9)(d`7B@WW2Uz5&#AvwP=kCs!MW*c&;g2=6kq7y^8zPqe;2!SPKpP=#6s_G1OzblJ'
    'TUSfVqK<-<^B|PwuH(Ln&7qxRJ`gB=7|*KxM>KWz!>c-b+VjK-OIlR;7341+X}C@eUMOPD10^=8^8~6W6)~4QLx|dz5jr914Ch;2Ll'
    'oAVz}we@bnM<t`04uy52n=Phs>2U`oaw?nWzjhiFYCL;54$wK?LMa){-?V;?d^*C(PbkhGq>66rbq9?-EDJGZjrZeV~bG$oi5$vSZA'
    'T(?#6t)#u2vP1Eu1gPwnWR!grMr_tLx&SEfTp!?NhMBGdP@2{=k5<*QOX7Ci9vPlL^0~j)#ca|$Yu$koi*Tc9hJr1v2w{t?x#~B~#+'
    'nnItAl%~phem9y!XQ=vO>|Ol`K1z~tttt|svP?{yq}7|5LI|90o#kMKu{=#p6$F!qD%II&G&9pxq1v5HK&8t;0Q=CS=8XY98LaD2L{'
    'LyYMJPRW^5e0V&*yA8J`RToGQ$Deguy0jzYhc$KcxPa3UUhg>P;ZfES4(aMU4;cKFG#2}z65t*Qc^q;JReu0*_)CJFsk$s}U+RJ!iq'
    'T~5htJlvR9%Ve>ZTw*FunnLp$UPw>Co#r;=qNF9t>^nj0Yd7HvW+o{1WaAd)d0^bZg35O<@@)5U(k2pNEp)>Ur%%g7`|rwhqV8D|-B'
    '}I&4<uP}wOo+Aan?F@bPDYpeaX$RO^09Ww&B)C)#Ot|AZ^{3PH$BAQ0MdUD5X$~9eb*|@s4ps`Effq9+gIi&T_o*%mT%RSnB3J1*fi'
    '#huS44NYtjgcuo2tYckoHR{SZTy~^?UH8p|E=^LVlMW@jl3bSCzi}N(@y#&=?e;MNh>S0CKS^kmHL!3rWQNd7v>ZQ+^yI+pc?=R};o'
    'zW2}Flr<prdfkzwJ!czq=-ugelTkctiZ|T1eH9ZOz)PD<K0a%`TtB$Bh~1})hg4`ZTJx?zR$t?(>?I{t#nNJT8MQo_TgcbOsL)w2r}'
    '=J@Z)I>-cI2wRO*OZ2PX>CH2*hLLUtC>-I!08oC{~xh6=;#3Paq#$C$nKdj&bBIvGan7vj|ay6}&Q5)2M(rVndR)5ZBwG<Z~#FEq-;'
    '*fK5BBbkC5HBE7M>N?u?v4d${gJ|C7&SZbvgum;;_^!m~C^n!3cBKq`x3LyV7fyi4ojN$<*l~1RD9zu~{z&$3c}427RPmFn7AOBSnE'
    '&`d9m6Df_@AOP@y6=w;&_>b5+PCMND)oqxo1BjAxfgCG)PiulwWg^l39dMhLS`jQc0e>_f2I;p^)Z5rFl?Eq~Y!T2hO_Z+`aer^I7Y'
    'xb>>7ZqH{wM;NgJ_lveVY!K@W*aNcgv+9@VK%_#Or&y0NTchVr$J6L%28M+t`5_?--MIrPkC^ZfS|K&k&|Nc$<b+jLx4c4ICOENUvT'
    '1xEjpEnH*?FXqhZX$GXOzh)<rCyWyc5xzgd3ED#mQDA59%9?%Ph!QgV(8b=$+<W`=lV~a2bb9sW~j+w+tz4y-^782oAoD;2r*5x*T?'
    '=%J-G1lQrh?96I9om<94@v`d+e#d}~#OgROsX94hgDk8QDosgG#V^*XS=VkUW-JOH;l3;``e2hwzKp~cV6!RT$}tkXxA%nQfRlTo^G'
    'Y1&t8X^cb7Fm2d$qMpqnf!zr(#NMfA(8SJ;*z|E=tMk0_MUf1BUvL(D3#U`Faxz8^?Gw5DnoI}W#!@6Evs3qSsr^qhH1%_57lx(4gt'
    '`Le9T*L5a!)Wtvm64atcL3~pW&vZ{H5vZf%2tCbCscA@mbd#xP2f6#PTaSQQlDMtf=O2kTEyJ-3Yp$E`olIQ=zDBHLR)DrpA6-+4X_'
    '|Tsh%9hTOQq$8GXq3G*z`-&qTai*iw}uTot0Di4x|Uc~pxRh;A7Z_G>46!P}_z%Z9IaF}8NI(^2hB;6D8@5FHd;$%K)tO=|sy2<Vff'
    'I3#QA-}f)GEy@!%IFd#t4f2$oV9GDvIo=K^&D(GkFk9fMKs?{gZ7?t<TjWLr|NzMAe-q<aRKK+#WGd&tDuSBzfMNHy2XrDOv~hot@S'
    'A}d9Zl7d?aN0l|m3{(CENp?4i~VUiPys><bQrZ5v<k_p7$EJiTGe?&T(Gv+pO4IJJU}Rr?OLr8EA`>p84{#4>7ke@G{{{gcB~Rq@U0'
    'O6a#oMV$Lfohw=p$DZzZ#bVbr<7asZ^iAFLk9l9g>dbCbk-3H~kCxExv9YZF!)AJZpp4~2tH6(hcGfL7lGDEtNa<RaDe!g*Zt<K;hi'
    '|Gv&yW`0@@^w(e^3^~&?6LKxRja<Y(Z~`7UN$#L&ZWf^jonDoW=3HV~Ibvp~{3mVR9T+qzo5YC;k6@$9mql;tt<j>`jvma#(3d7S8X'
    '<0#n7_$}R&-aOjAqWe04q;%p4It(*<Lfiw7osyG-EHB-E@U?q%KdC#5pOkm#{1x)#N6ik$iDf_FA*f63ItkTz^vU($)D0YSi`PFpVy'
    '1)1&4uYoR`dC_Xio$lCW62%r;#V*Fk-OwI7ac3`XARYbDK@kD_d{bjYm4z@73)j0;=izJw>r4=!hx=~h0}yX*_^fXT`JpsjWW-yhOk'
    'I?PVdfY2$Y*Am^9_Gf|xzbyfdGkTucT!6$9&fqs4E9wRp@U2^&f-N;Yd`fZLFR>{Nl8c%7;_o;VW%?Nh$tjMQi<_W6sa<}dr_l^smb'
    'JBDvc>oK+I13Yf7W_d4DAmYn?y65l0t}SgK`(0DWQIyCZe-}lY7cLjC>I@J*tk{4Zsd8Yo_Z+Sm?@#AjwedsJRoJlA6y4t1f|0R1ss'
    'Ee;UuIWO=f<J5Q_7g+-E3&+Jyl`wmQ>O(lTb<bMmTYE8#g3>7i6gIBO}WjU}Jv^*S4yFg70+7@_mPMY~z`my_DGVh%@H-+=i!lYLq)'
    'hiOKAFjx!#t#VFl_=p<)=R|fTm!JaR1lWinjG}DJPbvIW0Z~dTUf0)zdBJ!Ix0`v{^@Rr1$U7KqTW98LpR<kKB$U4NiOZ;%Ev^AueD'
    '^g#+33Mq5OmWs1$wRt`hh92?@%6=cSoRAnd{`yg_)&ll`;|#?K{IPTSu5&mn@;SF1?FvPg~6@K5d2$5EM;bho5DVG-v%pi##W=jc6@'
    '&@8aIt>en)|a!8+P4tH3GT89`0*?O3|68s8-EVTD_j_`3m7?8rb<K@o40Vd69@pdIMY6-G06Ut{gF_w$plB+<O@jo@*)6X(5?1?LlH'
    'crd<_-MsP$Ki6onov%D0U|JI-4YDG`JF_6G{0BRgSWm}m576<hx7?_;3sGs=Eu8k@Uu`D%(zl9mSlxM(=eG92692InNX?it$c&lE7U'
    '6p9WbEr&NTpu6*zR)y%@@|-$Etx8k`Rthd#|C4<_#R~YmDh8M{uh~5a!DKvU|`ziB=8mMRU!^7@llG*PCT=hE)&Fau>6dDl@LVkl_}'
    'qDL5i4g?}+LoB38Jpo(V_%hJ8d!jn&+SHx5PTJUrfl_^rKnH3s3snDq$WlZM`Ff7-V`MyX(t($p#K)+;E+@8r#F=)ki+VYb2b8(`!?'
    'h-V3x&{w~?L#G-e$?51l<91h;zZ6-IG|0Z(mgUjv}KqnDs3)Dz0KMDlHC{3c-tuE<f29HBO`FjhP&MQ8$gL(lex=I>rmFdh##^kS7f'
    'o|4C;@1$c6Pzz=p;G{PP~U{`R$a^hr6Mdq02*2lD)=@#`?fJ(m61JD%!uALE1RqnL3)9Je>;CBJ1*p=3ynKZ|#Gg`dydMwz>#Y4@GO'
    'Sm5q035h5`)sUB1sb^2K1`Vev-BB=jNGZGeDv%v3EF%L0#JK1i5J=zQov|WuPrZf0(EY-&Pb$Lt#bvDNm_ONzp0LkHYvD=iKFHOW`_'
    'JR$gzWZZ<mlVWo=%h!0=KEdq@9`k1LuB1-Jk{#4L->|uFhqjL*$_Dn>YLNX8{a*m<N~NjDnwQ@8Z#2=@2_oMywH7fki%kFtMh9Dz7Q'
    'Qi`zTsnDcbHxYrnlJr0AIb~DaB?;hCfG2mWX83_OBKB6}*`ou3dPs&aG#LFi3=h9vrhb<elVSA1aoVitkRaqv&qv8<eTNXtyKN!~U='
    'O8O~GI_3_El36}qOgq`c=yUCiuryC|8w4k#zQmMkRV@peA*cg>utkT)6`(krYv!FeF`)1Yi1RDGojd#$AN|mX}=Dm_PB6>h>3#q?Kx'
    'C3>^=(ad#L^M9<jyWaWwv#6J(YzhJd&0QTOx~+*f#mc`7AS&4v;5b{m7j>XpzjD4ANOTGAY&RMeaOj9EnAgRX{r7}r=0cEc}F)od*`'
    'X|)gRQ9BQ1J@YCrpZd&}9M}RA#>~eNeP8}@dy6EXPYV<^?ZlRa?r>CY8vZqkq48B`_+-t0_4HE%ZdM<|S%+%rtjQ+usc2(Q@@LSQ-}'
    'gD&Q+HU6Q!f+0TuURW{vu`faoM#GF!!_xq(%P%eDDz-OxFjMk21iIG^C678nI}1CCBlHNo$NcRDG?4L^*A7?}TwQv$~Jve7?Y1?xv8'
    '{k0oUK^dgg_rBhUg3ux=c(Zq)Dytcj@70lCt>B%qXgUU(H+0g)91~qft&a25HzX#Q4Mbp%=y`stfuW9MyOYGJm2YiG%6f5i(@0T`Y`'
    '<B-cXs^S_tU`YNxMW(9I8pLg#t-GI{P1Ah5IAGn3&#2Zm{xO8nD=-d1vPw=M2;FohZm>gPguuozIF!l>(eBMb~w|7_Ze*Af6>_3Jr7'
    '_Vg3SVTm>8_a6kX);Wq>M-`gRjNdgrpQKB3fiZ89IAs!H+Nv#`wZHCn_jCsaBEt-%G@a=RF1&7H7X=Lv5z&5~xR7*pWSw|w|#efk~$'
    'mF3)1qn;&!?ES(eVAixxG$dJpUO42zjyh?owDzZ0W8U&>-Rk)lv)oBz=^D|x<gM^^*<!r+^%RY<%9f0Mk%DLQ6ToYzH+p;uWpliaVg'
    'F-Mpl7p(?XOx1#rHK~bWIOi^EDLjd+}J~tc+DVE8ud5H|Q2rK)+`mqAFSm(k+QtFAfJ&1rcyfW5M&pHS}NBh4wov;c-nLH!6G-h>h3'
    '71?fa~Zp|bN4%m#3vx2z|My7D|aw&QEsexY1JLVm7hfQ}?V%J@?VACxV>i^V{9h|Do{ki`UZ4!Igk(fMuVSifOI>Z_3S`P9hrcvy0K'
    'V|3*3ZXCWGGNM?K34lDf^|(lhK0Ycz`dN)xUA?CGq_?T?l?S(9yq*^JUG9HPWwHh?_2sYg`&gYIYbpAkNw5-3Eq&N<w|Qmyn*RIzp<'
    'A`-&4Fv3VhxyC44V_Kt;L))9zNGk}C%TuUd*DL)^%-v4WO4l(7L1Rmp4rd8%F&2aakM5Wh!OkV)B&sw(BAs5gXN`0$m%H!q<*dp#Vh'
    'jihIri^!>J4b-3gOg$NqASLHdiK=$ok8%Bk%*APBTQNXr^SA-3bwJNA`h&AsHtOyRVGGZ#6yK69f{Sa_AwhN}rSI~9fw%f|VJGvLto'
    ';wxHFE%Ye_1LhPcFl~Crrhg2DrdQy``WvYM5|ujv9r@g>sj=u0r0}Au#>@5%gJM0d6H11-IH(L9_8SJKQ0H@gYZ0w=Ej??|DqKZWY3'
    'CeV(RU4HTwK-y?34zQta9IE$m+7z-wLy?9wV9de_tP`r8+f5!G0Kennr@J-L@-QUf0^L7@fEm31rhbEAtn=ZXF3ByTO?D#cCp5lr-o'
    'v{1qFLv5ywAiiUCTp}g&sin%Sh#PXc$@SRR=P-mN>A6)^aCSM<>d|<86_*$&ys`h`n#yxPaiwxG{A1(N1&lJ7nWT-%ugBR3_;`NsZH'
    'z&dA*wA>Y#@-g0p1v!qaJ{Y$gWIy2>g)ay0dEo}~D`o3MRBf-vplW^&d%!Orzl5qI28WvNeBFxLqa$ffiWao=BI&b(?cI%myyY;dNa'
    'kUbP|=O~)w|6<2p<WtMh+wenf3GueNaOSrZ{r10repBTk`uIWEnUjI*%gWJv#Q+=}XDPN@mJIJJN<l$$ARP-A_&SHl+-h$nf=jFL&{'
    'bRV+ZzF|b!CKN^LRGEQXN3gfkl6A1d(4NEIW0zGSM!NAUhXc%jr|>$}enfcpkBsVt!8BGj>hwI(snr0;G%UP;dEDTo{Vv<KWNh*xaU'
    '5!JhQCD+7Ds2a})7v;5PtRM7PW`x-xk+^7=JQJqWmnPMS$lnHq|6iW_2bHMJkjE!pxB5f^qnEEc6#jms`-;!l;Lfakg1Ra82yJ;+N='
    'n3xS=RB$^zd;Lh_eo;=hy}&}2exKKWc)^zd@P^ZJ)C%irOsM{o;~~cMQ@Vn$8&vVaN;!9OQo?WjY)`Cw?m-eM^SONGUd-t;lfw#hJs'
    'LS%>3_>=$g)U=!|p6ryl>jV@U)X>s(3O9X7J?gkflMWfYy?Ck>WQIwblH<Ke-{Bx)0+VM0SIrg(8+TAmIa8u1XZ_$GJP?HXOab%pH-'
    '^B~2*NwB+h2Mo2G!}g6E&VTW9#jP2wwvo$CVNIB{Sg!0S3}hYvQ=Q0n<i8Mq(u7Sf6Ci8za$1&C&qQ(^XjWehUgP!1d2o{Wj`?oj^C'
    's|%C*+Ha9`H=*{80R`Z9ELNFT>@_rTI_ZUSy-N9-!D#@D2+m;|K9vaJVO&50w^w@O=e)Tn33JE^A@_g$H4Is~KH6vI|t<7q&W@v4yf'
    'u<lU1fx>Q`ow5le+H<f5GwY3xf8Rvp}quTkS+#-+)?&PP;F&A(57jxbI&pD&@6WPU`vxR3VQn>4NASO)s#YY!rz^jUjT-w9sxKy)%w'
    '^(WnEzzebpmi==;4u$M!ux|&*Fdb4N~c-QOWEsD1zhaF7&swqP5Lr25ZV$A5{sW8f3ck(URR8I$F`%!?tDC(Q@|P%<w!R1GUhCCh2w'
    'XPNM$mEHEOBk|JDs3pE*NW@!3$PREd9;=h5nErnFf_o5FrcbDmKe-1S9z{I##ZS=_#Ws}2mHmIt0<t-SLXtkb}1))c~HzYYA6^E#w0'
    '-UwF>{zyKimhx4)`>;ct%|;HZC(o>8;p-!y3pQh!xyD$QW);WQy$!+DIhOQN;S~!}&tOZkB%&b`Uoe$C8+7kCgDu-phuS4aP~mPF+j'
    'dj~m*y7Z<w-}dq#=q85{BUC`B$;@eJS6&Z3=2XK8rdN9Z=V7H;e5yVymJiGu_ux+%RiL9KLAHzZy+Kx#9bf`%#Ds1}v9s5zS&F|FQX'
    'yF$NOlqz>MC(OLej$^cfKH4yoaYIsi84R=f5L#<tTDErX~4_vrmXA_o!3k)?_w9jGu=9J7VUv=OF`-N<o$3cvGzZHwVzeU#)0S{fdh'
    'I?-5Nsjm>Fyqe?S?^pArZZzbXWGl*nZx7pz*ZS-(){O_Gx|8sl?v!pv<^r0oD-SO8o<k?O=R^|2k`6c3g#2Y(7C0DJGlHTd-h1gM^3'
    '25uipjaiVv{OZgn`{LLH}SY-B1eMVRk#i5*iht@PWllGE8Zkk8sPU-bIcSazUQf@@cu;+FbY<HP5rEbQtIws4OXc1_S=`9+g?D_bY#'
    '_g)4^|4u~pIU|_M?+2Xbbe=tV8AsVi+8MqyrfyFS@wyS7VxP&S;-m6qT$$_=@sWc`unV-na+nrf&i@JBB}YI%-kpNZ){&L`58U*2v>'
    '-L{2HYB}3vtauA@kN7dLUy3?aIgDk?%=7xxADeE(ym`DfcOO&?zu6>w<j>U9@P*alGPlmX<s*qfsGyxe424g$R?McwJ2*)Sa*q?x~x'
    'I&HiRkrrt2|wu-63NK11zQ+7V>GIb*Dqxblu6OF}6-JxQg+Z)85346fXbT2n_bc;yg@*@nGpeQa#+)i%O8bRT`v=DB3hF)FvXFoqQL'
    'vd|2de5z;qgSUu@m@u7rsY^DaCLyeDo?2v$6-V9S|R<`K^DKzj+MH7ptZl%>~r1vs9rpnP9KY)lFD)%c}tbbz73&?17AbKuxL6NHJ('
    '4}D29^wr*KSJEEK%6M#F<0B$+-0U;g;V8lM)^)CbS0V9!<1l?jEq;6Y+(7ZJZ|a~RC}6V6WcAC0jAIxs@vH}$(WpS_V8iaN!!dG69>'
    '81ecU?%Z~Wt!~~==bZPk?#24BW09u#b&fLN`~n;_zJ`gGXb5MV20~r#CP><qBeJrMriW1lG)dyXb|!u2FJAn`T$_$z_1$Cm^7=qrmu'
    'rTp4Zk@)b}jz2%SDCar|f9uIK25t;C^mCPC{c8ef#m&E~eiZynSFC9#Y*0c^2E)EXP#*@7*z|Zks}b8kSJ2zAQOPDbxPW*>rqz7^kH'
    '%lGfzB;?%lxAg=Q}vkjcb?#+J3LLS8Ou_wFPS(h}hxL*rDZUxeCDK)exd5gOe6WG?z`(W$7hiqB=1$?7t!X;@mv7mo>ZhS17kK6E`P'
    'jCu`z=r@$`xD82#(v%~paZX8ZQ|uTy~*m#an^HYIks2bVU#(O?F+deUY<LYkBS~gBMie)I$ejw9oGi8mVIo(Lse*1vWDE5P7wBw!wt'
    '$-gY2V=R68~eLb(!>j(EV`@9+>`DcC2Dl|4>|ydmtEdYXQf#<6;3dEwKpO|W@`rucxXrx3LykTz*7=bwB(LI++4ibcwLAmeh3mTi1Y'
    'Ka#HD1vy9Y&&`W*{eywHb$1vZ-dPOqx*K8Jt`mI0%$YQI#3ycL@GwCsXaY@;TLAl}`U#3F6R2&?2;tG}B;lR5vG~v6YgACNN&MgVTF'
    '&k73;sl}m|`!;QCRd~(%&~md|>cA@gAuy;^lgSp*5nJ@3KyV7Wo69u(1<X<`xr5+d<flW-5poD-2V~#X~=G;Y80<>WwiGg4bm*jes>'
    '^IC7L&*nfa~5p$OQicjIGZw=&n#h8A*Tg&tmwlEf_XrG^Qfei1Cp|w^IarEj-IM4Shxqr97NjbyBu4RhCkKC7dca|<n&C|m2d;5eUu'
    '{jO<^8uedkA-{wt0-#40I2c!LsB)Rup{6lXeMfjgBSTj_$V2+_Uu<2Xfz!zPe`Ii9WG?AT*+3d%oS(+%cjm78zAGaK3p+9j(&r+X;W'
    '}9v+PEe_P|z9xbxf2wCoBE-@sGb)U#0N_JQe}-+;Sw$C91<b!s29o4QWUg!wZZDi3&kz~6(tkPcqs>-FU5<B*TsCr>Aw*cXlEi|0`D'
    'Jt<+4tuAI%Wn$lEPnu%1S+qYRigmfZVqe{~#c#c8NqR;M+i$&#mYPLV!URt??D8OSgVYXK>b_Q7?XV6qYxHpDwn%!TnaqTYFKlkWHX'
    'JiC4SHL>NTX^tY*?I4{rse1(6%h%Cf~#h!Kd(*#~bD*d+eW^?56LZgK4{&4<9W%SnyqXldb&!a~isOWIVA%JToR7lwRu6*vVN!XYP3'
    'B?k!Dc<4=NquR2w<sEKE~%EQpD-VjqBPWw~S1=l~>^f7BRiF2mXqotp4Ox-Q!vqXw!I{jucHcyyZ_y}?6xCqp7uizfYNWo64pU|hKN'
    'b#!wf$Kkhh^}-ICf-t@m0d4I3FoF!@TDLcmS6{)a_d<0z>)CHcPJKKo=XWS`mlL~yil;=5QRuJFm6Hw{~)CgKP&%$8w<WrYFRQBDtm'
    '}Jk_-671{O4TS2{d@bc6d(e2SX(_t2;9rr1wZgod+j<Eom))KFc?pQaY_Dv=YktQ}#`qnWh)W(Y*dU#IO$qtVe}Id{8jJW2g@Ct+6E'
    'Kjz*^(?X{6c?of}|C|M$&ejv3b*lms+Z32OX#fe!_rdVGHtY_SqkFAQOrhP2)~WT<3$rhHvzG_wb@FVwPXS9fBtg?37U+<*9wl32`I'
    '$Za;P2^PZnfnzzFH?=Xx9!D?#@n={Oat5M~kQ7*WzjHer-9mKDY_@S3Se_9jpH3feKVNhjX@}U%~fTKDp*bv&#D2(6p|C`pi<;R)tZ'
    't@E`MkT%AX*%4OWqs?or1pQ0<4vM|-b5W}uKM5}M*<ZqaP#r6>>Gx9EW&Kn_gH_Onyr#r#Edp>^5Ze?SFZ0W!x2^d{gg6~SoxaVgw@'
    'N&b!x=9`~l`Syhf-ZOQAW`d|Yh*X*4|D#noGJHSVW-wA2|aIZksJJ!K8{%ljY24|w@eECJIvrog)XI~E5SH=E;-(lNHx8AaMwMKnI~'
    'OB-Mrf<=het6<%-w}$1R*^#td+(8%UQzA5htuH*8nEg7D<YMVNOu6%zt;X`Sm)Ox?N}-<dYBWmN)3NC7&H-7f0;PYX5%WJ3KxQ}C{J'
    'q(Y@Kl6m%&KlHPN48yJA=-6Fw@(YJLw$U87A&NF@>R{c$bxhP_4)SheVA$y2+><>AX};?$CKWN4+IwZ-g{?Mes9)kwMg=mLKWD&tg*'
    'mLtOAzn!zR5WG3|>dQh}u&ep}&R^RVg~M;D_~WH)n(zQ^(QbcuQuPnF2pvpJzI@<y5$4nQ%R~gTFM%n=L-;AeM~y&X%0;rsCB)LY;a'
    'C_i5N{nsKu~IDH%^@cC!Kt9~l$UllB58OXu1CJWeN8-ddsyoB{q9;o?6SMY9%qxxKb@^o~<i0>h=##Z&;e!fsxOAm(Yx>xYoH>q}%H'
    'TnDKfuh!OsIFZ|nhDd`wWa_1W}cLI%aajcbF~G()SSSFJ9fc>2j+OTq(2or{{h*uGuh~wt1&moSl~Xo;N(a-LHm0!TO{%q{Pe#`tio'
    'ME_YKeLGR>i6L<DSC86x;q7K4J9C8!(y0uvuu`1V9b$UQus5<>cLrJS0$&sYlVBk!@$U;eZ>Po0bxB0Ka`5Aa!t_}g1QaY|tw`)+rg'
    'F6|%AUfPu7Hp3k5zkMg^#5fK9syLQ>UU|axy%D(W&;{@xc?TxkTugGA>bQ2RC(Kuy%-E1MuxiRv47+6k<C}}%M*pYWe)-qD>daK?mR'
    'NB$>!b1EKUaq}vS=JS5?b*YR((^VzOdgIm+r(ydn=OEo*nR1aR9!3GgWX}T!fLS%CuBr9`*lk3)IhS!K2?IA^$@QuJb+!X$JA|g1-a'
    '1KDP1eT-qzI-<1}x*s%`|nr}giXa3Z=V=g54&-v#i_i&qC7IV*@OW#~>v$u*`!aCg-G+*ivRXod=SkLW&jwhb*WONI!HGPn<+F&^)c'
    '$!ns;o;&Bmmb2+Cz~NSR1Oy9z2kMdw$R@7jkI`4D)YsB+8@~qPF~Km`Kv6<%TI@8)7oHxLImkQYT^`&*MOO(HTN|1Fr56@#twd%NM)'
    'J_`40h(qUEWX+&$GY{1W~fXAg=OYrb@%QTuI4J-3R&+TQR-zuo{fhe>2sT*$QGI;o#YWUD*QW1GfFUYceL%DF&FE8R)r>;wD0Il!hD'
    '5qNpszub~?!}i6KVQk@M)b7+~s>TCR_t<rAc+z2XOIG8jH>I)SmPg#;7F*W%EsfjiI}rPgIKy+suW_Wy09r0>#4cWFLhYQb7+R@@&g'
    't@0($Il>3^-gIBa13MH*ogSdQ@z_i-Xq0;-?^UZf^Np(Ix4Z5_?ZW)~%b3`<WMB+xVVSO3Y<wW5@G~clpNoa?Bz{k=p<KKq)&*yqI5'
    'xEhAD;dgvn@=UBsUu~f(FABRd-hlU^*D?yXSBi!})5<GQ<=e93AReAQ~98??|$2@jh^MOS*STpuB_u+jlzhlu8Zq=|Ne6H~tGlgGV;'
    'y7p2DgVnIpZS+xmnqM7Yb(+dkpY@NDB}+FJBs!4XPM6C0TTV-ffCob*ICAI32Iw?;Y+5;b2`_faQwVNEVtavUYN*{s@_CY7`;}+Efa'
    '}K3+r*9MJkqs1h7$KGO+1C74EumAwS@%A+pKp)VRHkU!?n+%gKwyH#hfU?)HB<>E*}8{dK_S4m?a>Yaw28Qjf3k833DIqA9U!B3<_E'
    '55LPZC@^z9)c@H8PMS_|@_HHFR4+&SO*t@aT@9L*4Tq`nz0`9i6Z<VX%sj_All<RiTy`J}4=nM3x#NG*>W2>e82b~PsBtG*k2!)p`i'
    'F4DiUBlf+y%1!T7hF!71*k!chKit49&Wj4vn)5Y0H5Tbg23a_HWxq+5cw2w3Y^Xyy0KXtaPSN3s-~AhdLTO@(A8lX@X?E2+B5Bz?-%'
    'GSoq7EtYLd9ggQRwC}S@q<z3}>4x7)GD;BfayOW@E#6h^K@e1dkHlaW94We^@eBgRb51w0D#V^w~fP~%>$(__={LSM_VL)~%3x4bc4'
    '^o?H;$I``*BS+v9<8ERy~kPA-S^C9{2n@$Y6^SZ7jx=8BPsd%ADFl=3Eqgk#FcY*a=TX<(Wx0zL1Ft)YK#2WizlZ-zM&rbaj%bxJ!G'
    'kTVk8W_qKzwDx1;lY#L2l%O!?_%yrQAR9J*uJ*IiF>s!j{4TK^I4En0%2nDgka@50W6zD1ia6Ywb4g86@pxWv$vyiR;D_p3>fJy$K`'
    'AzK}_4PqJpV~;3gqgZlz^)Y-N*o%J~>UfV+k@)vc2X|}8bX@UDp8vJ?2#dPC8M`v;xh@+={$RK%JH5byeS1*AcfIyyd0!v$e}`*{0{'
    'q-Dy5%V9b|f;dW2IPBdkbI9e93PuNaoy|cky{=WY|{c0A_Fu*vTC$v15udGwt7>Wh_cUqn>iUD*ivo!>rl-@FRb?J-M;0+36H}cJP>'
    ';(%wGKK2e38T$7IK(ZF`@ZkM=fWpRIMF5sh>9Ht{*$k}}Q$D6(lxG+)!wQfvB)lm{wm-~--W)DU?@A<gbz6KSGH*f*l{V?}*{=e)7?'
    'v;EzJ~`3HXFV7o`cxZ@olCOuL`fnx9kI3BvH2c8*r>(0*NH58=w&=Qa3gH^a2_+&`jPgD64IZrom{^!`*+5=+%^LTF5y-_`X1T@ht_'
    '+M?%)0b^+i+M2q`gcO{d)1x_IcO9DUBdM(-9W;rM&QsmfFZEM`~;t<g-_l60FJS2z|u-kO5X=ht|kwTH6j|KeK*Mnit+6+Cx$4$T>#'
    '4IdU95iR!2;OBaY;9cYq3W=Lftr2qE{E$ehZ99#s_ZqqSheqtz1xs>>+Yf6WXJXal*Z3kM28@Ospcg}9S@xf=ob0aSv})fDmi%`TJ*'
    '(dg!`mH%xP}c-aG?<!a`thJ?!VZJ0C$>fVnj(gUs1_|r_xD*Fvr!51syg5#~Iexx51AatZfh0yE1VwlYtON4S3a42#5Xq!$g@b>TJ$'
    'pUE(%0xwsie_l^`C=WSy@oi!=4{vgCHm<|1p6^L)dFM$U!ais5PfOq|qB$H!~!>}o;kX2(K%07GtpOk6{R#rn`%*ng#_r%38KCS@XI'
    '`pSTM`N*H{645i+ej@Z+wsa@qHllpLZ170inQs^2Fm$R$HEBkaO{FB4^2qVUIrG;dQ6K;D*0dEr@+9@GPrwx1oiCd$G$d=z~5tygw2'
    'K(_$FDF(UEs}ac(s8N>qbox@!1$yn?W1@GTZxQ;bg)W8mxghgetW$j*4=W0~@9>UcI6GUgZ2_U(T$s5pYt{jWdsNmK{FW@B*5IE)(_'
    '{{zv{0!V#vhVr&vBkhsdT*eM-2=kT}FMb*aZs&ZV=k;impH|NAXf7nhukvhrX()RW9Zpfz0YqE&ilP%VAvtd%`@LfZE{-c<_vQ}6H|'
    'k6A^@&exo6Th|qA(fkPUZ<?kHt~ww+7yHryFh=asU@tyu^6(>!cVn1Gc7_Gp%1+$Y+q8IM4F{owErQr;3_!rOigF4*$(M;(|z_bQ&#'
    'Mu#x#(8Y>R`ae`h}-+|=PlaTh+UG#GF47PFUPCi*$!q2pq#x{coT-W6`9CcidZhcDPKa~`tR%9#g<i_DJ@f_B9sEvCWq>Z6hmH&Nb0'
    '2`#K!X>R;%8Ccg$GCMb(Q<V%8XQ~2TdaP>k3ahzlO)7y0-j*Abfn$#HN!chruY1Di%!n)?r(nQ`BHo&eHi=J1mgCf0W^J1H`nRp!v6'
    '3lv8vUNCs#Rsc%U0Ux8f<94c>sO6RlaS;}h;erV*Q@W6%6;C-GJ<ese=rzVIRb!z;aI)6p}}o_Q+1;D4nL;nxQ~L#MtZZu*1@oEYPc'
    'DI+_1t0Xtkf3p8cE)VkJn}5b&byzhXXK%RM28p<4VitZ;jYIQ)1KB#Qiks3g0d3<>F(|x&m)aum+H^avJN6de=;qAc&&c2xEliUXeO'
    'QE*hpurlF^QP?aXhoVvXw2;E5!PjU3mJ@ZoG8XOO&A<hvmCI^6Qs4iJI3(^K%DH#?)RdEO)d8WotWF^L`sVUfYX`vux?lzyvnOOBzB'
    'nve~2Dhw!$w1yA)EQM$)STAbBMR{0V*wd)8}P0qvP4p!{>z`<bXI7nc<iA>*FmwlaX&2_f~(ATg5V)3{-9I(|FZzxYfKk-tKs4mAn3'
    'v;06TLzuJagdE^pG+Gy2at|JvuIF!FE+1j6Lls$!E0G{?5(*Ov_^(glC}&S@e^2e)h1RLR>~H*ErOo4ZD?_G3Mlv9qpdee!E5S8+M<'
    '1uts0_2bj1U{FR<Ws&l$n4hCqmEcZauohmhTDckmmrNHpcfeV8}ViR>Hq(*grK+V{Z)jz`C1bg%;?Saq;it53LRa4~zYe+&)pC4<QN'
    '7h65y7zhzMu<DRDg*Oc22WW8gFv$^wamL_Z6vrMG52DgtZJ-k!K~L747hg6RhC`)2Dd@pX-eYPBwqPZTJm&=dW2D8R-@g2g_sP8K)G'
    '&D1QpOG(>tX#%bzn%c4Q`%vk*t24rI(yMEVkdrnn(T-#UI=TW29FKlesadS$7t1l^jLZf4KPZ+Xim#;N6ttdIKJmPJ(T>RH3sug{cc'
    'S*bLKq^lhl6Fgf}NTsJGh$<cPO>vAR>x^t4;3$wT+12M#XYDKTL!vyPFf&8F;u43_wvy@SLgU?(&htiWh;Y!3jT6_6CoqwE8%2`X`x'
    'T33_^WgcSlGXFU?$HFHU&to%Tu?^uXTJfXKZ9_{Ye&(}*m=zVvpdfJGnb7#Xbqmf6`&}qjqx$Mg5or9SgJOgT^rVpMxm!6<n2?onpa'
    '>NX9ttgV_$lhUdvwimtu|(N$oR@VXN|cJE@*{Hm39*$vn9VF4wlh6<ZCE54OexTOS%e;XS@G`a|!R+(miIx1^d<0{T}rQ-8Z!vCB|3'
    'mYlVe5(gh*GhRN%CGjU=!V@D{b@K$QI@!m3r#uD8RCUM?n+1!quEOlvQQ|?TCbFju{}|uBlt#uJ6bio>!=sot%(c0VDJ{E;X09pl@{'
    '~05vP^{!_mW9=4bZdmBKYn5lZLrA)Az_zY@b9Q?k#s^G50H>c)m6EX7_NvV%*rpW#c6?o}Ysq2ZMlJy2pn8W8B>abC~~kWqNOSnSKV'
    'z(##Lp|7P@Icr>L6eWRxff9q$!lQvuWHl`T%b}uKRWh*6q#o;tO<05@|l}_(Xx53034H~L+nibd#rosJ_+1~vt#L$GWWyBB|*)N!L?'
    'TduXb-O{w)CrO;Qo%FFpDKes(moSq_PkqGtnnj~Y3vMx9k)-?$K9zC7ya36(!c$Ev3&!l(_EOn<r+4{q*!qAbQWe_kAd*d!||nu9*g'
    '+86S9})(xZMEY*7x-8G|(1y+umgay<xO@+L^ER1|I<j7M&!BV_MarQYG?{OGxX&>@+~*1jD`-^04-<L}q7y+&3zqMQ%6`rKgjXIm<Q'
    '9q_io8$1+hslg(ZKmIq7R)3PF=vou1n3;`l<khG%_&)lk|AewBJE^<QOwz7Yga-yo&@HEkI}nsZT~pHVUO)<}{rbpaH|AjT6B~ZqwO'
    'Q;(zZ@F*^*59Hb%wiL`joD(nu3?MJwd5q|92T(-p;j`=Rah#Zz*l;t;=y#EBIddDCH9lQL&-Xm(-ba_+~!G`Y4V%codF}{D5CQ!=T0'
    'SGL{9WgH3w|^ww1}8zW!%U7rf456$N%RWD(mc8VaZUn}D;9m5=N9@Z?j0P8!?xKZ`p)c2{5h33`q7yC|;pKrNn;e02EpMMCC?w$x0Q'
    '?DZL;fWnr4cS$PIJzRNBF&k{S?1QW{G<&&Xb`J{(MOzU(NQnBKG>Wojd+Wq(g>DtUl#MYA!37@qxqsmhdG<oXHi>m3@1I&hW4~(!qi'
    'c-NM!GTwU%m>zM=>sM2<Lgi7`duSX5bXil3DAk)0S<4(k>3;OXKTrfAO~$4i=2md0aq#RitP!yS^Xzi>z9yHWFfE1H=2o4;okiR~7)'
    ';`u9iPI1zEUT0P~EALwYmjge6iK+_Q=#zqtEfeVUL<1ar%@VT4$wGF!Hq9PT&cgCkAkobnyqbf+VS};g$T&GVQFD#0I@5u+ZQJ3}*2'
    '~P?_&R$xbr2b<-C*Y}eYll<ag_9ZC9}`J%~Y?X!jz@8pq(p6oxgH91r`M(A6PQ`j|Pz9FoPoVYGB`4AHMt3M?NZ1l@7ly<J7k`qHK!'
    '`w_=|qnB9uU3ClNdZ`mYzzAKu%D<3hF&pB*It1{bMeV*I=ubJv*Sc#|f70^yiPZm1-7wgHq$qD^N;GfxQ6lpF)me130M&n4kJEP;cE'
    'vkDUr<e(8EtRxtLK)|C;vjqRZbYT&30rRVY&nvA%EK*YHtb&B7fwEP8_tt=!Oh~8>_44s)R}kyzj)W+m(U6nUppa6_n(5>2G;Xt&0F'
    'wMn*+aE)(aODPG$UYK+Ru$eD3RsY>#lB4Tla^@TZ4gz2pOTak(DKTKe$O$G!L^&zGV`(h_tzz*q!p7PZchVLiS3P_oe)w^j6^Y0O?$'
    'E|ZCAD@t)qeFV2oI~ZMjaxlO9GWVwaCB6}PvU-yUHh%mjmU!G&GS%Q9ORGg3@pb_#*>8@g#V2t$UFF6;-pQ`q@Wqjs#Iz<w@uLow;G'
    'O3&cwx&W_VoKDw#rhPT@OuV3v*-9X6#99ySyHa(kt+H^b8c#m6>6d18*Svf$Mytgp21CvHy<kmKYiZvBc?_?Bn;7{ASxV?A-2m+}};'
    '%IHhzN|4d^F_B4lZ)qj>@^&2b9F!sjlEBg`GUe0<~y|VK+?UI}lne$4imZ-<qVUL+Fw{_ABly}R*uzYEHEBh6X#5S<`y*ljJfYY=vc'
    'K}sfY~V$jncSH_{h5wRB=hLk1$u{-Fyz%PxDno;CQ2Kz`S&NV2dalL>X`+baQ!h(es`EOMd#Qv<8)SV!C9<+t4|c=JBjjw0!5J;>M%'
    'WdI<GRvmFmxgu=lFAsNE?mv;~+~>f9a9N9O;)n}%oEf{Iqwmyt;6VJ=kIQOm7dw-J{q&q3YBP7LTQ1)nWXn4NAvYRfpmJK{9vKXEQ*'
    'k1yr#`zgb6qobmm@($wL>h`csa-Q^FY2oCeDe(MC5WBhQDC<aDLz5L``I~S5a5rbxveA>J;F8xNwr<@4?3Hb=^h;g_DdG+M?wA06>!'
    '<=c8WPSu3e@JhLpcad_{ySuPKqSMPhh~*AK+S)L{FrQ*{JO05V`+2{*}GK(yCtLL+Jx-$kxkPIPV)ytuu$m&sU*}o2%eAp_x4O&+~h'
    '?=|bOhtUUX80@bJA<DRa+PIw&H!o?Mw@qQ&(>SRE&3OdZx?Jll%?N8-(V#wLu$GRIgGRaaYh+4E44@;YoWqA;5outThdFWwCw-H^~-'
    'cG^cm)M};H4>*cmRK9%2~B?MDJ1O%&X*a8cEg)kb|d5E?i<4Pk{{^SZHTvS{9-2KwWuVig0^L!V@otY;O*j(c(X*nB%Noh^mY&KF34'
    'k@1>Nl4&=;c34I-F%-VV~<iusot`@y#L>Fi70cN!eJf*c!Ez{PDo{5i0SZ7&!I^A`lul<52Hub(UBjM+&|_ik`fMpE!9>^8fucY)th'
    'v4vL~_X?+(ngOS|fn96eDLQvH5G><*nO{qPs9Ya`6X++}S>0#uGdH5b(?a4uH?Y6AV>r_#k&<%Ft*GrIMQsL?n9QOYG?p%Bk`mc}++'
    'tlZX|Fcv@yFTF6cxPlN0Gdb^XRZ74>b>XkjJz^EZ?LS2Nf@(#mcd)GpJKEChIP{usRw~b=5LS&mEXB`!6gx)BumtBPz8Q8$s2708{#'
    'x(sJESI^efi7|=9F{A9hdApdPO+j@GJc!mD}F>{$GEVK`XZOa$2*5?|u!Agz2nKNE6k#S%<kJNzo`f)<?^?DL58U}d_uhaJtD}<4E;'
    '=#r11MPfyR&a{6hnkP(;&F?j?WX2`;kW~%AW~XJP+hP{NZN6nzHFE-Bz8%`j;=k>H)D*zj3<Z_qj$gp4-=t>v%_V1isEj`WEfr;4+8'
    '}qNH7^om+CrMsHufG;hMf+*zpOvmtTkBZvDh;`a_x?eO^dj{fzYd4+E}vNmqXrP+(RjYmRjQm6g7b_RsB~*}ukVvr1sBlDSaz_zpq+'
    'SC*irMY_?I@VMEYi*^bG!J&XGj5?s^J`XaBbcK6EF7nf!%TV|WcXs^O7`8XSS_tkOz{0Dg#24*d1qG=KxT60_l0;t@*se)}Hs4ONJA'
    'yzQ@`G=&f6T_$yn#dZ2jSkSV;K1H96lNt!CYUTW~T3^l0QtR^Wzr^?pu5ysy>R=`0Ybeozdbkk@-+reM%S|o&=+1IX0s;iU#U6vugo'
    'qP&PG-+dlj?_&in>MqLV}K0OXJIxU6AUb93Rt8PHW<^DoOSDaura~-(9pC&HbJenC_7)L{27BE>sTiDdB2sRGS$zezU$<K^r|7GWZx'
    '{DG$ZXW<M-VcY<1CU~D%bCmJ_wX|2BZ~-A0AcVIw(;s}LDIblW^LC2al==#3_1nn{svI)eH8XK7SZOIL(m)Ih`SHR!^mBi=u6&w+HO'
    '<;5lNPyeQFx$O}fRcSRx~gZ?J$N5tZa>D=j$u<8Jfzt2B0zFK<`uiC-Lc!Ku$$WFL;8^!fl?J}ef}@)Ak6U615yd3Z83S|Uoc5zao{'
    'DY;cQ0lwQy3F@!p*ow}x_%&5e*n0A}VD&?au4_z(^00MyBfy2n{{)ErHHhit+yc)ZM|jCuAGm!-hIO^6(YBBVRJ`dbjG0(XE2nADwK'
    'ogdSJ4Vca_lEgk<JjFjNK1=mz`rdkDGAb3vHO*ry#htoMNv=K4<k-hSXUvO#=qH!tUV#^m^k~DDyoJnckm-(AcfQnd_cl+3iHH#-75'
    'b#u3!gbBfjezD|!~-f>=^I!JrF_>#@kIZ#Lg!E$vt1!-NzrwyLaH>H65CMRR=*jcb(_)4ZHn8A{pQlNIhR9t!cJv&{W48!{}1?!$%P'
    '+#Fd>*^%@is!qjYe6%dS!7FNW+{QxOcA@<HI|B^10gIc29mn!*s|}g{J|s@iuP~E8R_=)@Tk6UStb-bdOXO|%9pu*yUu#1N{Oqc&7~'
    'WK^&m5HKg>D-!h<=ZN$&I+JiY{A-k4jWfwn)%V)YRAXlDVWpS?)s-3PGsnGX!HR~FiCMv|*TFSguv0`uZg>~;~dWnES9x#t3onXD|n'
    'U66~Wee>b!q8yf5R)asM1amM-OKA9@BV3ay!@P%T?Aym9Wa~W(HfD_!6x91eNQM;JFF!(eGc;j_?_WMl;VhQd8quAz@%)CF_xablBA'
    '8>bo_e+(fcwv)z+$<sP~2oLc-*OFH9u})mCbyXykj<vT+zgyEbk?Ck4(lVD$)Q?1r}7M4$HdBa7)D(wruM~ELa%CF7-_p*2h<}iu4f'
    'ltmtLin(~;u?QK>%BvLHa^%i+F7C>rizst+2tKsXSE421~2CdWkfn%TEhjsUVU`4qRD_(BTSg?S*R_ep;l48(ym9WOaV_00P2^o(NF'
    '&B%Y=$e=#l&LQR!AX@yw(2so=u@10<rA<iJ1NrW&S1@(9<o>EJ88&WMb7JN6(#(&fS{?-{OWJM^!mUtW}K%^(R%wR>aG>*ditDu*iV'
    'i=jWGw=5g`<Jpp%WBCL?+`zlG^_)-m&fZ8*^IC3dyMqvqfMh?wle*3CvXQ2h<l4@#g{7V?6;+g-^X!`0l%AK%G6r%V!R(#_Ybbmi)H'
    'Y_eNd5XEQDR)-T)4TaT<J49O(HR;pc_uQPdVm`p_4BP)BmRcQ0!n$BNwr=Tcc;fJbT0h@q)BB&M)5f8!=<yf4^|g!}=Hx|skEG%6?0'
    '8mb(ayY@9x{DzQ`$M{9R>|3$Hlug*^T<|J?;8j&2Fh?(>ybM=xxm5%D&xUN0z^2(hr|;Ri-ZZcb2DM)2WMHDJ>LdyNjx$2rs1Ugk#0'
    'dEN6!v7V^ogsJ<A#hZeG<RcRorxQ4B7l4S!j40y9Oo$P1u4PL|fnQ-MlbC{M8{qNteVY?pO<U&2PK>Fch{-jGKSFIz54-=f>L;qOr#'
    '>^kI1Y<;ZhHS;uLnA4EF7o4}zo2qh462>Y<oA08(9-2nG|t2lA_vuDY|&{pK2nNLG4A5C^L+4-{8GB36$`R+yd+h;A-7VlgMIOnh_>'
    '}^gvy$?IDNe=_&yl~!|uHny(th2RqrN|@~0zI_s*R9eQ{$}EAFwz;a}M9(^<j}Ee9sAX+xj)t3t=42qyWwiZaR_us?r-X%@_-W24r?'
    'u0OK$Ypw*3Oistxn=!~frtn>xRA^ya3$6>UW(v8suxH(8)|Q;j(x&X8Il6Hy{<JxnntWm%qEca~)NEp}GU(8Kj>$IlBhgY_SUlEHI6'
    'oi*>IRRZMcZD(AuD|*l8oW@9Ggkffm@i9`@aG4O%A&bAZAVMMA3~8NLy0zTc!^j&NpT;50~TjKgUV4JcT*x&y@Tctt)JMI}+pDySXF'
    'P$L%U#ufp(}Ogg_9x$k$ih42}On$=nG;HCmJ?aHBzE!Q!tY!CBVzJ>oXI~*^?BPeb>$6Xoyifg;4MxQdvMJv}XgmWi+*@;U=?7_nvr'
    'essdUEW?v-)cv(#nnIP*)&<6JUBAWbcUn%HNYat5S|qhw78prgVF;oU3(G>ahoaZzIBqeTBXu|dWY!VvjOO`ObZeMTKUQW!QdEqi~m'
    '}~a|Y=Q`hqkd#qU0*-hRktF33gKeHy~p!VK8kStEHft_a;`sZy<)A$p{>vfPA2IA!TYHe*K{_u;1qYE|#E!D^<s#P>hiDtXP_*E&ei'
    '=Q~kaLFV$bta6gfsV2*XHY5yc!DYX4M7A=k1o?P7TAw#WxbJWr%;XbEBlarux$}&DQNF=msUF9*bLL5Uy7F<+hI05d;SW3H)s0(sZs'
    'HEkJcC#N$}`*JzoFpvd#)hxAiLA|8r+5#lJl2p?&`|{Y-h0}^Xst@mIp3nLG~aRu0M#J#d1gpTFfr1?#0^a*;MpXgY_NEW(MEnDhns'
    '<C24*sOuNy**CqFHrz54oux&8JdF#QelmKp`lngfA(Z}YKOgMBcomK={3C33CeBuO6VQxz;iK3Js`{;C$;gQ>5Z8;WVH>)!<VJfSA{'
    'f5#OC<|_}|G&SnpCngLvDd58*yGw5dVlE~q+M`hHTLF0;ay*-YSjWY{cWUHZwzM>I2sW!U+^Cl1@>2JVF}%!;yVdK&haZ;HdBWCO+|'
    'Fp+X@V(D6nTs^Fhn<AkztI!(jQr6dn4S;jK8@(wauk>z}bd*G6EmlLh@ZD2zUCUqhcN{$O&&28ykzrD-Z5f_Y~p+}(19zKxBA>sO7%'
    'c^dE8$mW#)HC#8|RG)DGz{7C3m(3zB79(@QmSS@5{XO5G!bC=KmT*_*rLB>ecFwwvjgXalin*6*l(%NeOQXEq`#fLl#mm8vx3-}!J7'
    'mV$S&QMY{)x{yA9m|m76|8On6@$)WWSsR98j>9jjL>8*RU!(Wl>U?zmv%gMmW*DCB!T}1Nm!7XdP<}|Gx0re(MX;%EzK!=PS@%5{ur'
    'ILZTBiMUJo5soG<oL$0nT^_SnF3E%|su`4t&)r9^qDut>=Ln7YT3+7tewDV6r*y?rwo#q;<$zK;6G>FvE0}2c}S1;+f*TT{d=16Qt6'
    'L4<88M&u!;AcKb#A&+xmi~LZIBuMbWfqeUsFeL=aUKU_m*ID3HCuNqrADS-(KGscaXIY)ObBu%H9sfOCCL$(lx=hU)on}_Zx1DDrtG'
    '551q`1y0AG*O{6Ui;=31>}BA0T~8(IL<6G}ETdK*F$Dru;b8F)oZk&w+g7Ug3N>T^xRZHFmH8taMO_?NszUl$Z;JeX%p12eXYVaDs1'
    '@cyF!euu6Q2RHI@?A9I7>-(ODNpi_W1*)o3{Gqr}!^}3N)T^*fo!I<7G<#iv<ccYFsxSwgijFcnshk*IzDsPyo2<Mz2Zj%2z+6u{t|'
    '|OLG`*$fs$`_!{Td2~=eP*xava|0j+W^j+}!vvptpDNuGPP?nume(X<RS3+oTb3kvBa*o(;A?mXRA<$01_pVR*^WfID#A7syNs%L+e'
    'AVhT>MF6}vV8ER$GBMs_>$IYC7(p@~P!=YJ%7JR&Fg){SF{==D%;FCRxz&$)f*?;M3>*W#D+!yj99~JwobWD9im<-?KX`}hIFmS&rQ'
    'mfWoadxx@91g`nq@6up=y2pan~mVNNBgk*h(6quEBSY$q%6|=Rg9}@;y$ha9E|sGa1*k>A<6L+kuC`M>wdP7S)hX<zDAHNuO+9?%#s'
    'roLJa37AT?t*Uzs&azI9Qt-oieX`$r_UM6DC<Z4Xprx!|S8GE!+X13rqC*2U#1NpJK!U=`wnvfx_yuJj3gQe4Z%H7^e1up8Ay6(}eZ'
    'U{hUycx`YBGRxjVrV|FyTEXyFz6nIH<U@P79%fn}pp&I4d?UOHzJ5IdPtL!Ig3y1|tBV#;X%NH*l;~o7+a6jHsDzun9Yl~>0p{^aNa'
    '@O^dHwg`<jXxU5a&pBd%TFFyq)c@XOgAu1!g&xO1B#ykeuNaaxCLH@oXF<3!*yW)~Vsndem`-Lu)jzD3JXRSzG-{'
)


def _read_weights():
    raw = zlib.decompress(base64.b85decode(''.join(_WEIGHTS_B85)))
    n = len(raw) // 4
    flat = struct.unpack('<' + str(n) + 'f', raw)
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
