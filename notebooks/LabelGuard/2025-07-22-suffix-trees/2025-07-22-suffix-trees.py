from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict
from bs4 import BeautifulSoup, NavigableString
import re

# Приклад
ref_elements = [
 """(AL) "LOVITA" BISKOTA KLASIKE", BISKOTA ME PIKA ÇOKOLLADE (30%)"""
,"""(AM) Թխվածքաբլիթ  «Լովիտա կլասիկ քուքիս» ջնարակի կտորներով"""
,"""(AZ) MİNA TİKƏLƏRİ İLƏ ŞİRNİLİ PEÇENYE  “LOVITA” KLASSİK KUKİS”"""
,"""(BA) „LOVITA“, KEKS SA KOMADIĆIMA GLAZURE (30%)."""
,"""(BG) "LOVITA" CLASSIC COOKIES", БИСКВИТИ С ПАРЧЕНЦА КАКАОВА ГЛАЗУРА"""
,"""(GE)  ორცხობილა «ლოვიტა» კლასიკ ქუქის»  სარკალას ნატეხებით."""
,"""(KZ) ГЛАЗУРЬ ТІЛІМДЕРІ БАР "LOVITA" CLASSIC COOKIES" МАЙҚОСПА ПЕЧЕНЬЕСІ"""
,"""(LV) CEPUMI “LOVITA CLASSIC COOKIES” AR GLAZŪRAS GABALIŅIEM (30 %)"""
,"""(MK) "LOVITA" КЛАСИЧНИ БИСКВИТИ“, БИСКВИТИ СО КАПКИ ОД ЧОКОЛАДО (30%)"""
,"""(PL) “LOVITA CLASSIC COOKIES”, Ciastka z kawałkami polewy kakaowej (30%)."""
,"""(RO) "LOVITA" CLASSIC COOKIES", BISCUIȚI CU BUCĂȚI DE GLAZURĂ DE CIOCOLATĂ (30%)."""
,"""(RU) ПЕЧЕНЬЕ СДОБНОЕ "LOVITA" КЛАССИК КУКИС" С КУСОЧКАМИ ГЛАЗУРИ"""
,"""(UA) ПЕЧИВО ЗДОБНЕ "LOVITA" CLASSIC COOKIES" З КУСОЧКАМИ ГЛАЗУРІ"""
,"""<body>
	<p style="margin:12pt 0pt 12pt 0pt;"><span>&nbsp;<br/>Sk&#322;adniki: cukier, m&#261;ka </span><span style="font-weight:bold;">pszenna</span><span>, oleje ro&#347;linne (palmowy</span><span style="font-size:8pt;">&nbsp;</span><span>, z ziaren palmowych ca&#322;kowicie utwardzony, z ziaren palmowych), skrobia </span><span style="font-weight:bold;">pszenna</span><span>, kakao w proszku o obni&#380;onej zawarto&#347;ci t&#322;uszczu&nbsp;4%, </span><span style="font-weight:bold;">jaja </span><span>pasteryzowane w p&#322;ynie,</span><span style="font-weight:bold;"> mleko </span><span>w proszku odt&#322;uszczone, substancje spulchniaj&#261;ce: wodorow&#281;glan amonu, difosforan disodowy, wodorow&#281;glan sodu; s&oacute;l, emulgator: lecytyny (zawiera </span><span style="font-weight:bold;">soje</span><span>); aromat, regulator kwasowo&#347;ci: kwas mlekowy. </span></p><p style="text-align:justify;margin:0pt 0pt 10pt 0pt;line-height:1.15;"><span>Mo&#380;e zawiera&#263;: </span><span style="font-weight:bold;">orzeszki ziemne, orzechy laskowe, migda&#322;y, nasiona sezamu. </span></p><table border="0" cellspacing="0" cellpadding="0" width="718" style="border-collapse:collapse;">
		<tr style="height:8.6pt;">
			<td colspan="2" valign="top" style="width:307.65pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:1pt #000000 solid;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="text-align:justify;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;font-weight:bold;">Warto&#347;&#263; od&#380;ywcza produktu w 100 g</span></p></td></tr>
		<tr style="height:16.85pt;">
			<td valign="top" style="width:175.05pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">warto&#347;&#263; energetyczna </span></p></td><td valign="top" style="width:132.6pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:none;"><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">2097 kJ /501 kcal</span></p></td></tr>
		<tr style="height:25.35pt;">
			<td valign="top" style="width:175.05pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">t&#322;uszcz,</span></p><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">w tym kwasy t&#322;uszczowe nasycone</span></p></td><td valign="top" style="width:132.6pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:none;"><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">25,1 g</span></p><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">14,6 g</span></p></td></tr>
		<tr style="height:8.6pt;">
			<td valign="top" style="width:175.05pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">w&#281;glowodany,</span></p><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">w tym cukry</span></p></td><td valign="top" style="width:132.6pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:none;"><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">63,2 g</span></p><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">36,5 g</span></p></td></tr>
		<tr style="height:8.6pt;">
			<td valign="top" style="width:175.05pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">bia&#322;ko</span></p></td><td valign="top" style="width:132.6pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:none;"><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">4,3 g</span></p></td></tr>
		<tr style="height:8.05pt;">
			<td valign="top" style="width:175.05pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:1pt #000000 solid;"><p style="margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">s&oacute;l</span></p></td><td valign="top" style="width:132.6pt;padding:0pt 5.4pt 0pt 4.9pt;border-top:none;border-right:1pt #000000 solid;border-bottom:1pt #000000 solid;border-left:none;"><p style="text-align:right;margin:0pt 0pt 3pt 0pt;"><span style="background-color:#FFFF00;">0,63 g</span></p></td></tr>
	</table>
	<p style="text-align:justify;margin:0pt 0pt 10pt 0pt;line-height:1.15;"><span style="background-color:#FFFF00;">&nbsp;&nbsp;&nbsp; </span></p><p style="text-align:justify;margin:0pt 0pt 10pt 0pt;line-height:1.15;"><span style="background-color:#FFFF00;">&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;Najlepiej spo&#380;y&#263; przed:</span><span style="background-color:#FFFF00;font-size:8pt;">&nbsp;</span><span style="background-color:#FFFF00;"> [&hellip;]</span><span style="background-color:#FFFF00;font-size:8pt;">&nbsp;</span></p><p style="text-align:justify;margin:0pt 0pt 10pt 21pt;line-height:1.15;"><span style="background-color:#FFFF00;">Przechowywa&#263; w suchym i ch&#322;odnym miejscu.</span><span style="background-color:#FFFF00;font-size:8pt;">&nbsp;</span></p><p style="text-align:justify;margin:0pt 0pt 10pt 21pt;line-height:1.15;"><span style="background-color:#FFFF00;">masa netto [--] g</span><span style="background-color:#FFFF00;font-size:8pt;">&nbsp;</span></p><p style="text-align:justify;text-indent:21pt;margin:0pt 0pt 10pt 0pt;line-height:1.15;"><span style="background-color:#FFFF00;">Nr partii: [--] </span></p><p><span style="background-color:#FFFF00;font-family:Calibri;font-size:11pt;">Importer w Polsce: ROSHEN EUROPE Sp. z o.o., ul. Patriot&oacute;w 195/C2/C5, 04-858 Warszawa, Polska.</span><span style="background-color:#FFFF00;"> </span></p><p><a name="_com_1"></a><span>&nbsp;</span></p></body>"""
,"""<body>
	<p><span style="font-family:Arial;font-size:10pt;">Sastojci: &scaron;e&#263;er,</span><span style="font-family:Arial;font-size:10pt;font-weight:bold;"> p&scaron;eni&#269;no</span><span style="font-family:Arial;font-size:10pt;"> bra&scaron;no, biljne masti (nehidrogenizirano palmino ulje, potpuno hidrogenirano ulje palminih ko&scaron;tica, nehidrogenizirano ulje palminih ko&scaron;tica), </span><span style="font-family:Arial;font-size:10pt;font-weight:bold;">p&scaron;eni&#269;ni</span><span style="font-family:Arial;font-size:10pt;"> &scaron;krob, kakao prah smanjene masti 4%, pasterizirano teku&#263;e</span><span style="font-family:Arial;font-size:10pt;font-weight:bold;"> jaje</span><span style="font-family:Arial;font-size:10pt;">, obrano</span><span style="font-family:Arial;font-size:10pt;font-weight:bold;"> mlijeko</span><span style="font-family:Arial;font-size:10pt;"> u prahu, sredstva za rahljenje (amonijev hidrogen karbonat, dinatrijev difosfat, natrijev hidrogen karbonat), sol, emulgatori lecitini (sadr&#382;i </span><span style="font-family:Arial;font-size:10pt;font-weight:bold;">soju</span><span style="font-family:Arial;font-size:10pt;">), arome, regulator kiselosti mlije&#269;na kiselina.</span><span style="font-family:Arial;font-size:10pt;font-weight:bold;"> Mo&#382;e sadr&#382;avati kikiriki, lje&scaron;njake, bademe, sjemenke sezama.</span><span style="font-family:Arial;font-size:10pt;"> Prosje&#269;na hranjiva vrijednost za 100g proizvoda: Energija 2097kJ/501kcal, masti 25.1g, od kojih zasi&#263;ene masne kiseline 14.6g, ugljikohidrati 63.2g, od kojih &scaron;e&#263;eri 36.5g, proteini 4.3g, so 0,63g. &#268;uvati na 18&plusmn;5&deg;С i relativnoj vla&#382;nosti max 75%. Najbolje upotrijebiti do: datuma ozna&#269;enog na ambala&#382;i. Neto te&#382;ina: ozna&#269;ena na ambala&#382;i. Lot broj: ozna&#269;en na ambala&#382;i. Zemlja porijekla: Ukrajina. Proizvo&#273;a&#269;: PJSC &quot;Kyiv Confectionery Factory &quot;Roshen&quot;, 1 Nauki Ave, Kyiv, 03039, Ukraine, tel: 0-800-300-970, www.roshen.com. Uvoznik za BiH: Bingo d.o.o., ul.Bosanska poljana bb, 75000 Tuzla, Bosna i Hercegovina, tel/fax: +387 35 368 900/905, www.bingotuzla.ba, info@bingotuzla.ba.</span></p><p><a name="_dx_frag_StartFragment"></a><a name="_dx_frag_EndFragment"></a><span>&nbsp;</span></p></body>"""
,"""<body>
	<p><span>Склад: цукор, борошно </span><span style="font-weight:bold;">пшеничне</span><span>, жир рослинний (негідрогенізована пальмова олія, повністю гідрогенізована пальмоядрова олія, негідрогенізована пальмоядрова олія), крохмаль </span><span style="font-weight:bold;">пшеничний</span><span>, какао-порошок зі зниженим вмістом жиру 4%, меланж </span><span style="font-weight:bold;">яєчний</span><span>, </span><span style="font-weight:bold;">молоко</span><span> сухе знежирене, розпушувачі (гідрокарбонат амонію, дигідропірофосфат натрію, гідрокарбонат натрію), сіль, емульгатори лецитини (містить </span><span style="font-weight:bold;">сою</span><span>), ароматизатор, регулятор кислотності кислота молочна. Шматочки глазурі какаовмісної - 30%. </span><span style="font-weight:bold;">Може містити арахіс, фундук, мигдаль, кунжут. </span></p><p><span>*</span><span style="font-weight:bold;font-style:italic;">Класичне печиво</span></p><p><span>&nbsp;</span></p><p><span>&nbsp;</span></p><p><span>&nbsp;</span></p></body>"""
,"""<body>
	<p><span>Состав: сахар, мука </span><span style="font-weight:bold;">пшеничная</span><span>, жир растительный (негидрогенизированное пальмовое масло, полностью гидрогенизированное пальмоядровое масло, негидрогенизированное пальмоядровое масло), крахмал </span><span style="font-weight:bold;">пшеничный</span><span>, какао-порошок с пониженным содержанием жира 4%, меланж </span><span style="font-weight:bold;">яичный</span><span>, </span><span style="font-weight:bold;">молоко</span><span> сухое обезжиренное, разрыхлители (гидрокарбонат аммония, дигидропирофосфат натрия, гидрокарбонат натрия), соль, эмульгаторы лецитины (содержит </span><span style="font-weight:bold;">сою</span><span>), ароматизатор, регулятор кислотности кислота молочная. Кусочки глазури какаосодержащей - 30% </span><span style="font-weight:bold;">Может содержать арахис, фундук, миндаль, кунжут</span><span>.</span></p><p><span>&nbsp;*</span><span style="font-weight:bold;font-style:italic;">Классическое печенье</span></p><p><span>&nbsp;</span></p><p><span>&nbsp;</span></p></body>"""
,"""<body>&nbsp;<br>
<b><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1330;&#1377;&#1394;&#1377;&#1380;&#1408;&#1400;&#1410;&#1385;&#1397;&#1400;&#1410;&#1398;&#1384;&#1373;</span></b><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1399;&#1377;&#1412;&#1377;&#1408;, </span><b><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1409;&#1400;&#1408;&#1381;&#1398;&#1387;</span></b><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1377;&#1388;&#1397;&#1400;&#1410;&#1408;, </span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1378;&#1400;&#1410;&#1405;&#1377;&#1391;&#1377;&#1398; &#1395;&#1377;&#1408;&#1402;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> </span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">(</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1401;&#1392;&#1387;&#1380;&#1408;&#1400;&#1379;&#1381;&#1398;&#1377;&#1409;&#1406;&#1377;&#1390; &#1377;&#1408;&#1396;&#1377;&#1406;&#1381;&#1398;&#1400;&#1410; &#1397;&#1400;&#1410;&#1394;,
</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1377;&#1396;&#1378;&#1400;&#1394;&#1403;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1400;&#1406;&#1387;&#1398; </span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1392;&#1387;&#1380;&#1408;&#1400;&#1379;&#1381;&#1398;&#1377;&#1409;&#1406;&#1377;&#1390; &#1377;&#1408;&#1391;&#1377;&#1406;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1381;&#1398;&#1400;&#1410; &#1396;&#1387;&#1403;&#1400;&#1410;&#1391;&#1387;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1397;&#1400;&#1410;&#1394;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">, &#1401;&#1392;&#1387;&#1380;&#1408;&#1400;&#1379;&#1381;&#1398;&#1377;&#1409;&#1406;&#1377;&#1390; &#1377;&#1408;&#1396;&#1377;&#1406;&#1381;&#1398;&#1400;&#1410;
&#1396;&#1387;&#1403;&#1400;&#1410;&#1391;&#1387; &#1397;&#1400;&#1410;&#1394;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">), </span><b><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1409;&#1400;&#1408;&#1381;&#1398;&#1387;</span></b><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1413;&#1405;&#1388;&#1377;, </span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1409;&#1377;&#1390;&#1408; &#1397;&#1400;&#1410;&#1394;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1377;&#1397;&#1398;&#1400;&#1410;&#1385;&#1397;&#1377;&#1398;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1391;&#1377;&#1391;&#1377;&#1400;&#1397;&#1387; &#1411;&#1400;&#1399;&#1387; </span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">4</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">%,</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> <span lang="HY">&#1393;&#1406;&#1387; &#1396;&#1381;&#1388;&#1377;&#1398;&#1386;,
&#1397;&#1400;&#1410;&#1394;&#1377;&#1382;&#1381;&#1408;&#1390;&#1406;&#1377;&#1390;<b> &#1391;&#1377;&#1385;&#1387;</b> &#1411;&#1400;&#1399;&#1387;,&nbsp; </span></span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&nbsp;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1411;&#1389;&#1408;&#1381;&#1409;&#1400;&#1410;&#1409;&#1387;&#1401;&#1398;&#1381;&#1408;
(&#1377;&#1396;&#1400;&#1398;&#1387;&#1400;&#1410;&#1396;&#1387; &#1392;&#1387;&#1380;&#1408;&#1400;&#1391;&#1377;&#1408;&#1378;&#1400;&#1398;&#1377;&#1407;, &#1398;&#1377;&#1407;&#1408;&#1387;&#1400;&#1410;&#1396;&#1387; &#1392;&#1387;&#1380;&#1408;&#1400;&#1391;&#1377;&#1408;&#1378;&#1400;&#1398;&#1377;&#1407;, &#1398;&#1377;&#1407;&#1408;&#1387;&#1400;&#1410;&#1396;&#1387;
&#1381;&#1408;&#1391;&#1392;&#1387;&#1380;&#1408;&#1400;&#1402;&#1387;&#1408;&#1400;&#1414;&#1400;&#1405;&#1414;&#1377;&#1407;),</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> <span lang="HY">&#1377;&#1394;, &#1383;&#1396;&#1400;&#1410;&#1388;&#1379;&#1377;&#1407;&#1400;&#1408;&#1398;&#1381;&#1408;&#1373; &#1388;&#1381;&#1409;&#1387;&#1407;&#1387;&#1398;&#1398;&#1381;&#1408; </span></span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">(</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1402;&#1377;&#1408;&#1400;&#1410;&#1398;&#1377;&#1391;&#1400;&#1410;&#1396;
&#1383; <b>&#1405;&#1400;&#1397;&#1377;</b></span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">)</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">, &#1378;&#1400;&#1410;&#1408;&#1377;&#1406;&#1381;&#1407;&#1387;&#1401;, </span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1385;&#1385;&#1406;&#1377;&#1397;&#1398;&#1400;&#1410;&#1385;&#1397;&#1377;&#1398; &#1391;&#1377;&#1408;&#1379;&#1377;&#1406;&#1400;&#1408;&#1387;&#1401;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1373;</span><span style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;"> &#1391;&#1377;&#1385;&#1398;&#1377;&#1385;&#1385;&#1400;&#1410;</span><span lang="HY" style="font-size: 10.5pt; line-height: 107%; font-family: Sylfaen, serif; color: black; background-image: initial; background-position: initial; background-size: initial; background-repeat: initial; background-attachment: initial; background-origin: initial; background-clip: initial;">&#1417; &#1355;&#1398;&#1377;&#1408;&#1377;&#1391;&#1387; &#1391;&#1407;&#1400;&#1408;&#1398;&#1381;&#1408;&#1373;30%&#1417; <b>&#1343;&#1377;&#1408;&#1400;&#1394; &#1383; &#1402;&#1377;&#1408;&#1400;&#1410;&#1398;&#1377;&#1391;&#1381;&#1388; </b></span><b><span lang="HY" style="font-size:10.5pt;line-height:107%;font-family:&quot;Sylfaen&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-bidi-font-family:Arial;
mso-ansi-language:HY;mso-fareast-language:RU;mso-bidi-language:AR-SA">&#1379;&#1381;&#1407;&#1398;&#1377;&#1398;&#1400;&#1410;&#1399;,
&#1402;&#1398;&#1380;&#1400;&#1410;&#1391;, &#1398;&#1400;&#1410;&#1399;, &#1412;&#1400;&#1410;&#1398;&#1403;&#1400;&#1410;&#1385;&#1417;</span></b><span lang="HY" style="font-size:12.0pt;
line-height:107%;font-family:&quot;Sylfaen&quot;,serif;mso-fareast-font-family:&quot;Times New Roman&quot;;
mso-bidi-font-family:&quot;Times New Roman&quot;;mso-ansi-language:HY;mso-fareast-language:
RU;mso-bidi-language:AR-SA"> </span><span lang="HY" style="font-size:10.5pt;
line-height:107%;font-family:&quot;Sylfaen&quot;,serif;mso-fareast-font-family:&quot;Times New Roman&quot;;
mso-bidi-font-family:Arial;mso-ansi-language:HY;mso-fareast-language:RU;
mso-bidi-language:AR-SA">&#1357;&#1398;&#1398;&#1380;&#1377;&#1397;&#1387;&#1398; &#1377;&#1408;&#1386;&#1381;&#1412;&#1384; 100&#1379; &#1396;&#1385;&#1381;&#1408;&#1412;&#1400;&#1410;&#1396;&#1373; &#1405;&#1402;&#1387;&#1407;&#1377;&#1391;&#1400;&#1410;&#1409;&#1398;&#1381;&#1408;-4.3&#1379;,
&#1395;&#1377;&#1408;&#1402;&#1381;&#1408;-25.1&#1379;, &#1377;&#1390;&#1389;&#1377;&#1403;&#1408;&#1381;&#1408;-63.2&#1379;; &#1383;&#1398;&#1381;&#1408;&#1379;&#1381;&#1407;&#1387;&#1391; &#1377;&#1408;&#1386;&#1381;&#1412;&#1384; (&#1391;&#1377;&#1388;&#1400;&#1408;&#1387;&#1377;&#1391;&#1377;&#1398;&#1400;&#1410;&#1385;&#1397;&#1400;&#1410;&#1398;&#1384;)&#1373; 2097&#1391;&#1355;/501&#1391;&#1391;&#1377;&#1388;&#1417;
</span><span lang="HY" style="font-size:10.5pt;line-height:107%;font-family:&quot;Sylfaen&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-bidi-font-family:Calibri;
color:black;mso-ansi-language:HY;mso-fareast-language:RU;mso-bidi-language:
AR-SA">&#1354;&#1377;&#1392;&#1381;&#1388; (18±3) °С &#1403;&#1381;&#1408;&#1396;&#1377;&#1405;&#1407;&#1387;&#1395;&#1377;&#1398;&#1387; &#1415; 75%-&#1387;&#1409; &#1400;&#1401; &#1378;&#1377;&#1408;&#1393;&#1408; &#1413;&#1380;&#1387; &#1392;&#1377;&#1408;&#1377;&#1378;&#1381;&#1408;&#1377;&#1391;&#1377;&#1398;
&#1389;&#1400;&#1398;&#1377;&#1406;&#1400;&#1410;&#1385;&#1397;&#1377;&#1398; &#1402;&#1377;&#1397;&#1396;&#1377;&#1398;&#1398;&#1381;&#1408;&#1400;&#1410;&#1396;:&nbsp;</span></body>"""
,"""<body>&nbsp;<br><p class="MsoNormal" style="margin-bottom:0cm;line-height:normal"><b><span lang="LV" style="font-family:&quot;Times New Roman&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-ansi-language:LV;mso-bidi-language:
LV">Sast&#257;vda&#316;as:</span></b><span lang="LV" style="font-family:&quot;Times New Roman&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-ansi-language:LV;mso-bidi-language:
LV"> cukurs, <b><u>kvie&#353;u</u></b> milti, augu
e&#316;&#316;as (palmu e&#316;&#316;a, piln&#299;b&#257; hidrogen&#275;ta palmu kodolu e&#316;&#316;a, palmu kodolu e&#316;&#316;a), <b><u>kvie&#353;u</u></b> ciete, kakao pulveris ar
samazin&#257;tu tauku saturu 4&nbsp;%, <b><u>olu
baltumu un dzeltenumu</u></b> mais&#299;jums, <b><u>v&#257;jpiena</u></b>
pulveris, irdin&#257;t&#257;ji: amonija hidrog&#275;nkarbon&#257;ts, din&#257;trija difosf&#257;ts, n&#257;trija
bikarbon&#257;ts; s&#257;ls, emulgatori: lecit&#299;ni (satur <b><u>soju</u></b>), aromatiz&#275;t&#257;js, sk&#257;buma regul&#275;t&#257;js: piensk&#257;be.</span><b><span style="font-family:&quot;Times New Roman&quot;,serif;mso-ansi-language:LT"><o:p></o:p></span></b></p>

<p class="MsoNormal" style="margin-bottom:0cm;line-height:normal"><b><span lang="LV" style="font-family:&quot;Times New Roman&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-ansi-language:LV;mso-bidi-language:
LV">Produkts var satur&#275;t zemesriekstus, lazdu riekstus, mandeles un sezama
s&#275;klas.</span></b><b><span style="font-family:&quot;Times New Roman&quot;,serif;
mso-ansi-language:LT"><o:p></o:p></span></b></p>

<p class="MsoNormal" style="margin-bottom:0cm;line-height:normal"><span lang="LV" style="font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:&quot;Times New Roman&quot;;
mso-ansi-language:LV;mso-bidi-language:LV">Ener&#291;&#275;tisk&#257; un uzturv&#275;rt&#299;ba
100&nbsp;g: 2097&nbsp;kJ/501&nbsp;kcal; tauki&nbsp;– 25,1&nbsp;g, tostarp
pies&#257;tin&#257;t&#257;s tauksk&#257;bes&nbsp;– 14,6&nbsp;g; og&#316;hidr&#257;ti&nbsp;– 63,2&nbsp;g,
tostarp cukuri&nbsp;– 36,5&nbsp;g; olbaltumvielas&nbsp;– 4,3&nbsp;g;
s&#257;ls&nbsp;– 0,63&nbsp;g.</span><span style="font-family:&quot;Times New Roman&quot;,serif;
mso-ansi-language:LT"><o:p></o:p></span></p>

<span lang="LV" style="font-size:11.0pt;line-height:115%;font-family:&quot;Times New Roman&quot;,serif;
mso-fareast-font-family:&quot;Times New Roman&quot;;mso-ansi-language:LV;mso-fareast-language:
ZH-CN;mso-bidi-language:LV">Uzglab&#257;t 18±5&nbsp;°C temperat&#363;r&#257;, ja gaisa
relat&#299;vais mitrums nep&#257;rsniedz 75&nbsp;%.</span></body>"""
,"""<body>&nbsp;Ingrediente: zah&#259;r, f&#259;in&#259; de <b>gr&#226;u</b>, gr&#259;simi vegetale (ulei de palmier nehidrogenat, ulei de miez de palmier complet hidrogenat, ulei de miez de palmier nehidrogenat), amidon de <b>gr&#226;</b>u, pudr&#259; de cacao cu con&#539;inut redus de gr&#259;simi 4%, <b>ou </b>lichid pasteurizat, <b>lapte</b> praf degresat, agen&#539;i de cre&#537;tere (carbonat acid de amoniu, difosfat disodic, bicarbonat de sodiu), sare, emulgatori lecitine (con&#539;in <b>soia</b>), arome, regulator de aciditate acid lactic. <b>Poate con&#539;ine arahide, alune, migdale, semin&#539;e de susan.</b><div><br></div><div>A se p&#259;stra la temperatura de (18±5)°С &#537;i umiditatea relativ&#259; a aerului de 75% maxim.</div>
</body>"""
,"""<body><body>&#4312;&#4316;&#4306;&#4320;&#4308;&#4307;&#4312;&#4308;&#4316;&#4322;&#4308;&#4305;&#4312;: &#4328;&#4304;&#4325;&#4304;&#4320;&#4312;, <b>&#4334;&#4317;&#4320;&#4305;&#4314;&#4312;&#4321; </b>&#4324;&#4325;&#4309;&#4312;&#4314;&#4312;, &#4315;&#4330;&#4308;&#4316;&#4304;&#4320;&#4308;&#4323;&#4314;&#4312; &#4330;&#4334;&#4312;&#4315;&#4312; (&#4304;&#4320;&#4304;&#4336;&#4312;&#4307;&#4320;&#4317;&#4306;&#4308;&#4316;&#4312;&#4320;&#4308;&#4305;&#4323;&#4314;&#4312; &#4318;&#4304;&#4314;&#4315;&#4312;&#4321; &#4310;&#4308;&#4311;&#4312;, &#4321;&#4320;&#4323;&#4314;&#4304;&#4307; &#4336;&#4312;&#4307;&#4320;&#4317;&#4306;&#4308;&#4316;&#4312;&#4320;&#4308;&#4305;&#4323;&#4314;&#4312; &#4318;&#4304;&#4314;&#4315;&#4312;&#4321; &#4306;&#4323;&#4314;&#4312;&#4321; &#4310;&#4308;&#4311;&#4312;, &#4304;&#4320;&#4304;&#4336;&#4312;&#4307;&#4320;&#4317;&#4306;&#4308;&#4316;&#4312;&#4320;&#4308;&#4305;&#4323;&#4314;&#4312; &#4318;&#4304;&#4314;&#4315;&#4312;&#4321; &#4306;&#4323;&#4314;&#4312;&#4321; &#4310;&#4308;&#4311;&#4312;), <b>&#4334;&#4317;&#4320;&#4305;&#4314;&#4312;&#4321; </b>&#4321;&#4304;&#4334;&#4304;&#4315;&#4308;&#4305;&#4308;&#4314;&#4312;, &#4313;&#4304;&#4313;&#4304;&#4317;&#4321; &#4324;&#4334;&#4309;&#4316;&#4312;&#4314;&#4312; &#4330;&#4334;&#4312;&#4315;&#4312;&#4321; &#4307;&#4304;&#4305;&#4304;&#4314;&#4312; &#4328;&#4308;&#4315;&#4330;&#4309;&#4308;&#4314;&#4317;&#4305;&#4312;&#4311; 4%, <b>&#4313;&#4309;&#4308;&#4320;&#4330;&#4334;&#4312;&#4321; </b>&#4315;&#4308;&#4314;&#4304;&#4316;&#4319;&#4312;, &#4323;&#4330;&#4334;&#4312;&#4315;&#4317; <b>&#4320;&#4331;&#4312;&#4321; </b>&#4324;&#4334;&#4309;&#4316;&#4312;&#4314;&#4312;, &#4306;&#4304;&#4315;&#4304;&#4324;&#4334;&#4309;&#4312;&#4308;&#4320;&#4308;&#4305;&#4308;&#4314;&#4312; &#4313;&#4317;&#4315;&#4318;&#4317;&#4316;&#4308;&#4316;&#4322;&#4308;&#4305;&#4312; (&#4304;&#4315;&#4317;&#4316;&#4312;&#4323;&#4315;&#4312;&#4321; &#4336;&#4312;&#4307;&#4320;&#4317;&#4313;&#4304;&#4320;&#4305;&#4317;&#4316;&#4304;&#4322;&#4312;, &#4316;&#4304;&#4322;&#4320;&#4312;&#4323;&#4315;&#4312;&#4321; &#4307;&#4312;&#4336;&#4312;&#4307;&#4320;&#4317;&#4318;&#4312;&#4320;&#4317;&#4324;&#4317;&#4321;&#4324;&#4304;&#4322;&#4312;, &#4316;&#4304;&#4322;&#4320;&#4312;&#4323;&#4315;&#4312;&#4321; &#4336;&#4312;&#4307;&#4320;&#4317;&#4313;&#4304;&#4320;&#4305;&#4317;&#4316;&#4304;&#4322;&#4312;), &#4315;&#4304;&#4320;&#4312;&#4314;&#4312;, &#4308;&#4315;&#4323;&#4314;&#4306;&#4304;&#4322;&#4317;&#4320;&#4308;&#4305;&#4312; &#4314;&#4308;&#4330;&#4312;&#4322;&#4312;&#4316;&#4308;&#4305;&#4312; (&#4328;&#4308;&#4312;&#4330;&#4304;&#4309;&#4321; <b>&#4321;&#4317;&#4312;&#4317;&#4321;</b>), &#4304;&#4320;&#4317;&#4315;&#4304;&#4322;&#4312;&#4310;&#4304;&#4322;&#4317;&#4320;&#4312;, &#4315;&#4319;&#4304;&#4309;&#4312;&#4304;&#4316;&#4317;&#4305;&#4312;&#4321; &#4320;&#4308;&#4306;&#4323;&#4314;&#4304;&#4322;&#4317;&#4320;&#4312; &#4320;&#4331;&#4308;&#4315;&#4319;&#4304;&#4309;&#4304;. &#4313;&#4304;&#4313;&#4304;&#4317;&#4321; &#4328;&#4308;&#4315;&#4330;&#4309;&#4308;&#4314;&#4312; &#4315;&#4312;&#4316;&#4304;&#4316;&#4325;&#4320;&#4312;&#4321; &#4316;&#4304;&#4322;&#4308;&#4334;&#4308;&#4305;&#4312; - 30%. <b>&#4328;&#4308;&#4312;&#4331;&#4314;&#4308;&#4305;&#4304; &#4328;&#4308;&#4312;&#4330;&#4304;&#4309;&#4307;&#4308;&#4321; &#4304;&#4320;&#4304;&#4325;&#4312;&#4321;&#4321;, &#4311;&#4334;&#4312;&#4314;&#4321;, &#4316;&#4323;&#4328;&#4321;, &#4321;&#4308;&#4310;&#4304;&#4315;&#4312;&#4321; &#4315;&#4304;&#4320;&#4330;&#4309;&#4314;&#4308;&#4305;&#4321;.</b><div>* &#4313;&#4314;&#4304;&#4321;&#4312;&#4313;&#4323;&#4320;&#4312; &#4317;&#4320;&#4330;&#4334;&#4317;&#4305;&#4312;&#4314;&#4304;&nbsp;</div><div>&#4313;&#4309;&#4308;&#4305;&#4312;&#4311;&#4312; &#4326;&#4312;&#4320;&#4308;&#4305;&#4323;&#4314;&#4308;&#4305;&#4304; 100 &#4306;&#4320;&#4304;&#4315; &#4318;&#4320;&#4317;&#4307;&#4323;&#4325;&#4322;&#4310;&#4308;: &#4308;&#4316;&#4308;&#4320;&#4306;&#4308;&#4322;&#4312;&#4313;&#4323;&#4314;&#4312; &#4326;&#4312;&#4320;&#4308;&#4305;&#4323;&#4314;&#4308;&#4305;&#4304; – 2097 &#4313;&#4312;&#4314;&#4317;&#4335;&#4317;&#4323;&#4314;&#4312; (501 &#4313;&#4313;&#4304;&#4314;); &#4330;&#4334;&#4312;&#4315;&#4312; - 25.1 &#4306;&#4320;&#4304;&#4315;&#4312;; &#4315;&#4304;&#4311; &#4328;&#4317;&#4320;&#4312;&#4321; &#4316;&#4304;&#4335;&#4308;&#4320;&#4312; &#4330;&#4334;&#4312;&#4315;&#4317;&#4309;&#4304;&#4316;&#4312; &#4315;&#4319;&#4304;&#4309;&#4308;&#4305;&#4312; - 14.6 &#4306;&#4320;&#4304;&#4315;&#4312;; &#4316;&#4304;&#4334;&#4328;&#4312;&#4320;&#4332;&#4327;&#4314;&#4308;&#4305;&#4312; - 63.2 &#4306;&#4320;&#4304;&#4315;&#4312;; &#4315;&#4304;&#4311; &#4328;&#4317;&#4320;&#4312;&#4321; &#4328;&#4304;&#4325;&#4320;&#4308;&#4305;&#4312; – 36.5 &#4306;&#4320;&#4304;&#4315;&#4312;; &#4330;&#4312;&#4314;&#4304; - 4.3 &#4306;&#4320;&#4304;&#4315;&#4312;; &#4315;&#4304;&#4320;&#4312;&#4314;&#4312; - 0.63 &#4306;&#4320;&#4304;&#4315;&#4312;.</div><div>&#4312;&#4316;&#4304;&#4334;&#4308;&#4305;&#4304; (18±5)°С &#4322;&#4308;&#4315;&#4318;&#4308;&#4320;&#4304;&#4322;&#4323;&#4320;&#4312;&#4321;&#4304; &#4307;&#4304; &#4304;&#4320;&#4304;&#4323;&#4315;&#4308;&#4322;&#4308;&#4321; 75% &#4336;&#4304;&#4308;&#4320;&#4312;&#4321; &#4324;&#4304;&#4320;&#4307;&#4317;&#4305;&#4312;&#4311;&#4312; &#4322;&#4308;&#4316;&#4312;&#4304;&#4316;&#4317;&#4305;&#4312;&#4321; &#4318;&#4312;&#4320;&#4317;&#4305;&#4308;&#4305;&#4328;&#4312;.</div>
</body></body>"""
,"""<body><body>&nbsp; Съставки: захар,&nbsp;<b>пшенично</b> брашно, растителни мазнини (нехидрогенирано палмово масло, напълно хидрогенирано масло от палмови ядки, нехидрогенирано&nbsp;масло от палмови ядки), <b>пшенично</b> нишесте, какао на прах с ниско съдържание на мазнини 4%,&nbsp;<b>яйчен</b> меланж, обезмаслено <b>мляко</b> на прах, набухватели (амониев бикарбонат, натриев бикарбонат, динатриев дифосфат), сол,&nbsp;емулгатори лецитини (съдържат <b>соя</b>), ароматизант, регулатор на киселинността млечна киселина.<div>Парченца какаова глазура 30%</div><div><b>Може да съдържа фъстъци, лешници, бадеми, сусам.&nbsp;</b></div><div><b><i>*Класически бисквити&nbsp;&nbsp;</i></b></div><div>.Да се съхранява при (18 ± 5) ° С и относителна влажност на въздуха не по-висока от 75%.&nbsp;</div></body></body>"""
,"""<body><body>&nbsp;<br>
<span style="color: black; font-family: Arial, sans-serif;">&#1178;&#1201;рамы:</span><span style="color: black; font-family: Arial, sans-serif;">&nbsp;</span>

<span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
color:black;mso-ansi-language:KZ">&#1179;ант,</span><b><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;mso-ansi-language:
KZ"> бидай</span></b><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
mso-ansi-language:KZ"> &#1201;ны, &#1257;сімдік майы (сутектендірілмеген&nbsp; пальма майы,</span><span lang="KZ"> </span><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
mso-ansi-language:KZ">толы&#1179;тай сутектендірілген пальма д&#1241;ніні&#1187; майы,
сутектендірілмеген пальма д&#1241;ніні&#1187; майы),<b>
бидай</b> крахмалы,<span style="color:black"> майды&#1187; &#1179;&#1201;рамы т&#1257;мен какао-&#1201;нта&#1179; 4%,</span></span><span lang="KZ"> </span><b><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;color:black;
mso-ansi-language:KZ">ж&#1201;мырт&#1179;а</span></b><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
color:black;mso-ansi-language:KZ"> меланж,</span><span lang="KZ"> </span><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
color:black;mso-ansi-language:KZ">майы алын&#1171;ан &#1179;&#1201;р&#1171;а&#1179; <b>с&#1199;т</b>, &#1179;опсыт&#1179;ыштар (аммоний гидрокарбонаты, натрий
дигидропирофосфаты, натрий гидрокарбонаты), ас т&#1201;зы,</span><span lang="KZ"> </span><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
color:black;mso-ansi-language:KZ">эмульгаторлар лецитиндер (&#1179;&#1201;рамында соя бар),</span><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;mso-ansi-language:KZ"> х<span style="color:black">ошиістендіргіш, &#1179;ыш&#1179;ылды&#1179;ты реттеуіш с&#1199;т &#1179;ыш&#1179;ылы. &#1178;&#1201;рамында
какаосы бар</span></span><span lang="KZ" style="font-family:&quot;Times New Roman&quot;,serif;
color:black;mso-ansi-language:KZ"> </span><span lang="KZ" style="font-family:
&quot;Arial&quot;,sans-serif;mso-ansi-language:KZ">глазурь тілімдері -<span style="color:black">30%. </span><b>&#1178;&#1201;рамында
жержа&#1187;&#1171;а&#1179;, орман жа&#1187;&#1171;а&#1171;ы, бадам, к&#1199;нжіт болуы м&#1199;мкін. </b><o:p></o:p></span><br>

<b><span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;mso-ansi-language:KZ">*Классикалы&#1179;
печенье<o:p></o:p></span></b><br>

<span lang="KZ" style="font-family:&quot;Arial&quot;,sans-serif;
mso-ansi-language:KZ">&nbsp;</span><br>

<span lang="KZ" style="font-size:11.0pt;line-height:115%;font-family:&quot;Arial&quot;,sans-serif;
mso-fareast-font-family:Calibri;mso-ansi-language:KZ;mso-fareast-language:EN-US;
mso-bidi-language:AR-SA">&nbsp;(18±5)°С
температура мен 75%-дан жо&#1171;ары емес ауаны&#1187; салыстырмалы ыл&#1171;алдылы&#1171;ы жа&#1171;дайында
са&#1179;тау керек.</span></body></body>"""
,"""<body><body>P&#235;rb&#235;r&#235;sit: sheqer, miell <b>gruri</b>, yndyr&#235; vegjetale (vaj palme jo i hidrogjenizuar, vaj palme plot&#235;sisht i hidrogjenizuar, vaj palme jo i hidrogjenizuar), niseshte <b>gruri</b>, pluhur kakao me pak yndyr&#235; 4%, <b>vez&#235; </b>t&#235; l&#235;ngshme t&#235; pasterizuar, <b>qum&#235;sht </b>pluhur i skremuar, rrit agjent&#235; (hidrogjen karbonat amonit, difosfat dinatriumi, hidrogjen karbonat natriumi), krip&#235;, emulsifikues lecitina (p&#235;rmbajn&#235; <b>soje</b>), aromatizues, rregullues i aciditetit acid laktik. <b>Mund t&#235; p&#235;rmbaj&#235; kikirik&#235;, lajthi, bajame, fara susami.</b><div><b><br></b><span style="font-size: 14px;">Mbajeni n&#235; (18±5)°С dhe lag&#235;shtin&#235; relative t&#235; ajrit jo m&#235; shum&#235; se 75%.</span></div></body></body>"""
,"""<body><body>T&#601;rkibi: &#351;&#601;k&#601;r, <b>bu&#287;da</b> unu, bitki ya&#287;&#305; (hidrogenez&#601; olunmam&#305;&#351; palma ya&#287;&#305;, tamhidrogenez&#601; olunmu&#351; palma tumu ya&#287;&#305;, hidrogenez&#601; olunmam&#305;&#351; palma tumu ya&#287;&#305;),&nbsp; <b>bu&#287;da</b> ni&#351;astas&#305;, t&#601;rkibind&#601; ya&#287;&#305;n miqdar&#305; azald&#305;lm&#305;&#351; kakao tozu 4%, <b>yumurta</b> melanj&#305;, quru ya&#287;s&#305;z <b>s&#252;d</b>, yum&#351;ald&#305;c&#305;lar (ammonium hidrokarbonat, natrium dihidropirofosfat, natrium hidrokarbonat), duz, emulqatorlar lesitinl&#601;r (t&#601;rkibind&#601; <b>soya</b> var), aromatizator, tur&#351;uluq t&#601;nziml&#601;yicisi - s&#252;d tur&#351;usu. T&#601;rkibind&#601; kakao olan mina tik&#601;l&#601;ri – 30%. <b>T&#601;rkibind&#601; yerf&#305;nd&#305;&#287;&#305;, f&#305;nd&#305;q, badam, k&#252;nc&#252;t ola bil&#601;r.</b><div><b><i> *Klassik pe&#231;enye.</i></b>&nbsp;</div><div>100 q m&#601;hsulun qida d&#601;y&#601;ri: enerji d&#601;y&#601;ri – 2097 Kc (501 kkal); ya&#287;lar - 25,1 q; onlardan doymu&#351; ya&#287;lar - 14,6 q; karbohidratlar - 63,2 q; onlardan &#351;&#601;k&#601;rl&#601;r – 36,5 q; z&#252;lallar - 4,3 q; duz - 0,63 q.&nbsp;</div><div><br></div><div>(18±5)°С-d&#601;n &#231;ox olmayan temperaturda v&#601; 75%-d&#601;n &#231;ox olmayan nisbi r&#252;tub&#601;t &#351;&#601;raitind&#601; saxlay&#305;n.</div></body></body>"""
,"""<body><div><span style="font-size: 14px;">Состав: шеќер, <b>пченично </b>брашно, растителни масти (нехидрогенизирано палмино масло, целосно хидрогенизирано палмово јадро, нехидрогенизирано масло од палмово јадро), <b>пченичен </b>скроб, какао во прав со намалени масти 4%, пастеризирано течно <b>јајце</b>, обезмастено <b>млеко </b>во прав, подигање агенси (амониум хидроген карбонат, динатриум дифосфат, натриум хидроген карбонат), сол, емулгатори лецитини (содржат <b>соја</b>), ароми, регулатор на киселост млечна киселина. <b>Може да содржи кикирики, лешници, бадеми, сусам.</b></span><b>&nbsp;</b></div><div><span style="font-size: 14px;">&nbsp;</span></div><div><span style="font-size: 14px;">Да се чува на (18±5)°С и релативна влажност на воздухот не повеќе од 75%.</span></div>
</body>"""
,"""A se păstra la temperatura (18±3)°С și umiditatea relativă a aerului de maximum 75%"""
,"""İdxalçı: “Aqro-Vest DC” MMC, Azərbaycan res.,Bakı şəh., Sabunçu ray., Bakıxanov qəs., B.Bünyadov küç., 12. Tel: +994124255090."""
,"""Importator în România: ROSHEN ONE SRL, Blv. Tudor Vladimirescu, nr. 22, Green Gate, unitatea 5, Etaj 5, Sectorul 5, Bucuresti, România. Tel. +40761808972. / Importator în R. Moldova: SRL «ROSHEN SWEET», R. Moldova, mun. Chișinău, str. I. Zaikin, 10/6, tel.: +373 22238301."""
,"""Importer w Polsce: ROSHEN EUROPE Sp. z o.o., ul. Cybernetyki 10, 02-677 Warszawa, Polska."""
,"""Importues dhe distributor për Republiken e Kosovës: ALMA KOS SH.P.K. Adresa: Konjuh, Lipjan Kosove, Kontakti 044/442-022. e-mail: almakos2008@gmail.com."""
,"""Importuotojas ES:/ Importētājs ES:/ Maaletooja EL: UAB „Roshen Nord“, Svajonės g. 27, LT-94101, Klaipėda, LIETUVA. Tel.: +37052409524, e-mail: info@roshen-nord.lt"""
,"""Izcelsmes valsts: Ukraina."""
,"""Mbajeni në (18±3)°C dhe lagështinë relative të ajrit jo më shumë se 75%."""
,"""Przechowywać w suchym i chłodnym miejscu."""
,"""Ţara de origine: Ucraina."""
,"""Uzglabāšanas nosacījumi: uzglabāt (18±3)°C temperatūrā, ja gaisa relatīvais mitrums nepārsniedz 75%."""
,"""Vendi i origjinës: Ukrainë."""
,"""Wyprodukowano w: Ukraina."""
,"""Вносител за България: Рошен България ЕООД, бул. България 81Б, ет. 4, гр. София, 1404, България."""
,"""Да се съхранява при температура (18±3)°С и относителна влажност на въздуха не по-висока от 75%."""
,"""Да се чува на (18±3)°C и релативна влажност на воздухот не повеќе од 75%."""
,"""Зберігати за температури (18±3)°C і відносної вологості повітря не вище 75%."""
,"""Земја на потекло Украина."""
,"""Импортер в Республику Казахстан: ТОО "Кондитер-Азия", Республика Казахстан, г. Алматы, Жетысуский р-н, пр. Райымбека, 169, 050050, г. Алматы, а/я № 169. 
Қазақстан Республикасындағы импорттаушы: "Кондитер-Азия" ЖШС, Қазақстан Республикасы, Алматы қ., Жетісу ауданы, Райымбек даңғылы, 169, 050050, Алматы қ., а/ж № 169."""
,"""Імпортер в Україні: ДП „КК „Рошен”, пр-т Науки, 1, корп. 1, м. Київ, 03039, Україна. Лінія підтримки споживачів: 0-800-300-970."""
,"""Страна на произход: Украйна."""
,"""Увозник и дистрибутер за Северна Македонија: А-Инвест ДООЕЛ, ул. 20 бр. 80, Бразда, Чучер Сандево, Скопје. Тел: 070325800, e-mail: import@ainvest.com.mk."""
,"""Ներմուծող՝ «Դի-Դի-Թրեյդ» ՍՊԸ, ՀՀ, ք. Երևան, Ար․ Միկոյան 37/8։"""
,"""شركة روشن – أوكرانيا. بلد المنشأ : أوكرانيا"""
,"""დამზადებულია უკრაინაში."""
,"""ოფიციალური იმპორტიორი საქართველოში: შპს "როშენ ჯორჯია" საქართველო, ქ. თბილისი, სამგორის რაიონი, თენგიზ ჩანტლაძის ქ., N40/დიდი ლილო, თეთრი ხევის დასახლება, ტელ. +995 322 98 98 28."""
]

label_elements = ["""(RU) ПЕЧЕНЬЕ СДОБНОЕ "LOVITA" КЛАССИК КУКИС" С КУСОЧКАМИ ГЛАЗУРИ. Состав: сахар, мука пшеничная, жир растительный (негидрогенизированное пальмовое масло, полностью гидрогенизированное пальмоядровое масло, негидрогенизированное пальмоядровое масло), крахмал пшеничный, какао-порошок с пониженным содержанием жира 4%, меланж яичный, молоко сухое обезжиренное, разрыхлители (гидрокарбонат аммония, дигидропирофосфат натрия, гидрокарбонат натрия), соль, эмульгаторы лецитины (содержит сою), ароматизатор, регулятор кислотности кислота молочная. Кусочки глазури какаосодержащей – 30%. Может содержать арахис, фундук, миндаль, кунжут. Хранить при температуре (18±5)°С и относительной влажности не выше 75%. Импортер в Республику Казахстан: ТОО "Кондитер-Азия", Республика Казахстан, г. Алматы, Жетысуский р-н, пр. Райымбека, 169, 050050, г. Алматы, а/я № 169. """
,"""(KZ) ГЛАЗУРЬ ТІЛІМДЕРІ БАР "LOVITA" CLASSIC COOKIES" МАЙҚОСПА ПЕЧЕНЬЕСІ. Құрамы: қант, бидай ұны, өсімдік майы (сутектендірілмеген  пальма майы, толықтай сутектендірілген пальма дәнінің майы, сутектендірілмеген пальма дәнінің майы), бидай крахмалы, майдың құрамы төмен какао-ұнтақ 4%, жұмыртқа меланж, майы алынған құрғақ сүт, қопсытқыштар (аммоний гидрокарбонаты, натрий дигидропирофосфаты, натрий гидрокарбонаты), ас тұзы, эмульгаторлар лецитиндер (құрамында соя бар), хошиістендіргіш, қышқылдықты реттеуіш сүт қышқылы. Құрамында какаосы бар глазурь тілімдері – 30%. Құрамында жержаңғақ, орман жаңғағы, бадам, күнжіт болуы мүмкін. (18±5)°С температура мен 75%-дан жоғары емес ауаның салыстырмалы ылғалдылығы жағдайында сақтау керек. Қазақстан Республикасындағы импорттаушы: "Кондитер-Азия" ЖШС, Қазақстан Республикасы, Алматы қ., Жетісу ауданы, Райымбек даңғылы, 169, 050050, Алматы қ., а/ж № 169. """
,"""(AM) Թխվածքաբլիթ «Լովիտա կլասիկ քուքիս» ջնարակի կտորներով: Բաղադրությունը՝ շաքար, ցորենի ալյուր, բուսական ճարպ (չհիդրոգենացված արմավենու յուղ, ամբողջովին հիդրոգենացված արկավենու միջուկի յուղ, չհիդրոգենացված արմավենու միջուկի յուղ), ցորենի օսլա, ցածր յուղայնության կակաոյի փոշի 4%, ձվի մելանժ, յուղազերծված կաթի փոշի, փխրեցուցիչներ (ամոնիումի հիդրոկարբոնատ, նատրիումի հիդրոկարբոնատ, նատրիումի երկհիդրոպիրոֆոսֆատ), աղ, էմուլգատորներ՝ լեցիտիններ (պարունակում է սոյա), բուրավետիչ, թթվայնության կարգավորիչ՝ կաթնաթթու։ Ջնարակի կտորներ՝ 30%։ Կարող է պարունակել գետնանուշ, պնդուկ, նուշ, քունջութ։ Պահել (18±5)°С ջերմաստիճանի և 75%-ից ոչ բարձր օդի հարաբերական խոնավության պայմաններում: Հայաստանում Ներմուծող` «Դի-Դի-Թրեյդ» ՍՊԸ, ՀՀ, ք. Երևան, Ար. Միկոյան 37/8։ """
,"""(GE) ორცხობილა «ლოვიტა» კლასიკ ქუქის» სარკალას ნატეხებით. ინგრედიენტები: შაქარი, ხორბლის ფქვილი, მცენარეული ცხიმი (არაჰიდროგენირებული პალმის ზეთი, სრულად ჰიდროგენირებული პალმის გულის ზეთი, არაჰიდროგენირებული პალმის გულის ზეთი), ხორბლის სახამებელი, კაკაოს ფხვნილი ცხიმის დაბალი შემცველობით 4%, კვერცხის მელანჟი, უცხიმო რძის ფხვნილი, გამაფხვიერებელი კომპონენტები (ამონიუმის ჰიდროკარბონატი, ნატრიუმის დიჰიდროპიროფოსფატი, ნატრიუმის ჰიდროკარბონატი), მარილი, ემულგატორები ლეციტინები (შეიცავს სოიოს), არომატიზატორი, მჟავიანობის რეგულატორი რძემჟავა. კაკაოს შემცველი მინანქრის ნატეხები – 30%. შეიძლება შეიცავდეს არაქისს, თხილს, ნუშს, სეზამის მარცვლებს. ინახება (18±5)°С ტემპერატურისა და არაუმეტეს 75% ჰაერის ფარდობითი ტენიანობის პირობებში. ოფიციალური იმპორტიორი საქართველოში: შპს "როშენ ჯორჯია" საქართველო, ქ. თბილისი, სამგორის რაიონი, თენგიზ ჩანტლაძის ქ., N40/დიდი ლილო, თეთრი ხევის დასახლება, ტელ. +995 322 98 98 28. დამზადებულია უკრაინაში."""
,"""(AZ) MİNA TİKƏLƏRİ İLƏ ŞİRNİLİ PEÇENYE “LOVITA” KLASSİK KUKİS”. Tərkibi: şəkər, buğda unu, bitki yağı (hidrogenezə olunmamış palma yağı, tamhidrogenezə olunmuş palma tumu yağı, hidrogenezə olunmamış palma tumu yağı),  buğda nişastası, tərkibində yağın miqdarı azaldılmış kakao tozu 4%, yumurta melanjı, quru yağsız süd, yumşaldıcılar (ammonium hidrokarbonat, natrium dihidropirofosfat, natrium hidrokarbonat), duz, emulqatorlar lesitinlər (tərkibində soya var), aromatizator, turşuluq tənzimləyicisi – süd turşusu. Tərkibində kakao olan mina tikələri – 30%. Tərkibində yerfındığı, fındıq, badam, küncüt ola bilər. (18±5)°С-dən çox olmayan temperaturda və 75%-dən çox olmayan nisbi rütubət şəraitində saxlayın. İdxalçı: “Aqro-Vest DC” MMC, Azərbaycan res.,Bakı şəh., Sabunçu ray., Bakıxanov qəs., B. Bünyadov küç., 12. Tel: +994124255090. """
,"""(BA) „LOVITA“, KEKS SA KOMADIĆIMA GLAZURE (30%). Sastojci: šećer, pšenično brašno, biljne masti (nehidrogenizirano palmino ulje, potpuno hidrogenirano ulje palminih koštica, nehidrogenizirano ulje palminih koštica), pšenični škrob, kakao prah smanjene masti 4%, pasterizirano tekuće jaje, obrano mlijeko u prahu, sredstva za rahljenje (amonijev hidrogen karbonat, dinatrijev difosfat, natrijev hidrogen karbonat), sol, emulgatori lecitini (sadrži soju), arome, regulator kiselosti mliječna kiselina. Može sadržavati kikiriki, lješnjake, bademe, sjemenke sezama. Čuvati na 18±5°С i relativnoj vlažnosti max 75%. Uvoznik za BiH: Bingo d.o.o., ul. Bosanska poljana bb, 75000 Tuzla, Bosna i Hercegovina, tel./fax: +38735368900/905, www.bingotuzla.ba, info@bingotuzla.ba. Zemlja porijekla: Ukrajina. """
,"""(MK) "LOVITA" КЛАСИЧНИ БИСКВИТИ“, БИСКВИТИ СО КАПКИ ОД ЧОКОЛАДО (30%). Состав: шеќер, пченично брашно, растителни масти (нехидрогенизирано палмино масло, целосно хидрогенизирано палмово јадро, нехидрогенизирано масло од палмово јадро), пченичен скроб, какао во прав со намалени масти 4%, пастеризирано течно јајце, обезмастено млеко во прав, подигање агенси (амониум хидроген карбонат, динатриум дифосфат, натриум хидроген карбонат), сол, емулгатори лецитини (содржат соја), ароми, регулатор на киселост млечна киселина. Може да содржи кикирики, лешници, бадеми, сусам. Да се чува на (18±5)°С и релативна влажност на воздухот не повеќе од 75%. Увозник и дистрибутер за Северна Македонија: А-Инвест ДООЕЛ, ул. 20 бр. 80, Бразда, Чучер Сандево, Скопје. Тел.: 070325800, e-mail: import@ainvest.com.mk. Земја на потекло Украина. """
,"""(AL)(RMV)(RKS) "LOVITA" BISKOTA KLASIKE", BISKOTA ME PIKA ÇOKOLLADE (30%). Përbërësit: sheqer, miell gruri, yndyrë vegjetale (vaj palme jo i hidrogjenizuar, vaj palme plotësisht i hidrogjenizuar, vaj palme jo i hidrogjenizuar), niseshte gruri, pluhur kakao me pak yndyrë 4%, vezë të lëngshme të pasterizuar, qumësht pluhur i skremuar, rrit agjentë (hidrogjen karbonat amonit, difosfat dinatriumi, hidrogjen karbonat natriumi), kripë, emulsifikues lecitina (përmbajnë soje), aromatizues, rregullues i aciditetit acid laktik. Mund të përmbajë kikirikë, lajthi, bajame, fara susami. Mbajeni në (18±5)°С dhe lagështinë relative të ajrit jo më shumë se 75%. Importues dhe distributor për Shqipëri: Dauti Komerc SH.P.K. Adresa: fsh. Vrine ShenVlash 257/40 Zona Kadastrale 3385. Tel./mob: 067/208-8004. E-mail: infodurres@dauti.com.mk. Importues dhe Distributor për Maqedonisë e Veriut: A-Invest DOOEL, Rr. 20 nr. 80, Brazda, Çuçer Sandevo, Shkup. Tel.: 070325800, e-mail: import@ainvest.com.mk. Importues dhe distributor për Republiken e Kosovës: ALMA KOS SH.P.K. Adresa: Konjuh, Lipjan Kosove, Kontakti 044/442-022. e-mail: almakos2008@gmail.com. Vendi i origjinës: Ukrainë."""
,"""(BG) "LOVITA" CLASSIC COOKIES", БИСКВИТИ С ПАРЧЕНЦА КАКАОВА ГЛАЗУРА. Съставки: захар, пшенично брашно, растителни мазнини (нехидрогенирано палмово масло, напълно хидрогенирано масло от палмови ядки, нехидрогенирано масло от палмови ядки), пшенично нишесте, какао на прах с ниско съдържание на мазнини 4%, яйчен меланж, обезмаслено мляко на прах, набухватели (амониев бикарбонат, натриев бикарбонат, динатриев дифосфат), сол, емулгатори лецитини (съдържат соя), ароматизант, регулатор на киселинността млечна киселина. Парченца какаова глазура 30%. Може да съдържа фъстъци, лешници, бадеми, сусам. Да се съхранява при (18±5)°C и относителна влажност на въздуха не по-висока от 75%. Вносител за България: Рошен България ЕООД, бул. България 81Б, ет. 4, гр. София, 1404, България. Страна на произход: Украйна. """
,"""(PL) “LOVITA CLASSIC COOKIES”, Ciastka z kawałkami polewy kakaowej (30%). Składniki: cukier, mąka pszenna, oleje roślinne (palmowy, z ziaren palmowych całkowicie utwardzony, z ziaren palmowych), skrobia pszenna, kakao w proszku o obniżonej zawartości tłuszczu 4%, jaja pasteryzowane w płynie, mleko w proszku odtłuszczone, substancje spulchniające: wodorowęglan amonu, difosforan disodowy, wodorowęglan sodu; sól, emulgator: lecytyny (zawiera soje); aromat, regulator kwasowości: kwas mlekowy. Może zawierać: orzeszki ziemne, orzechy laskowe, migdały, nasiona sezamu. Przechowywać w suchym i chłodnym miejscu. Importer w Polsce: ROSHEN EUROPE Sp. z o.o., ul. Patriotów 195/C2/C5, 04-858 Warszawa, Polska. """
,"""(LV) CEPUMI “LOVITA CLASSIC COOKIES” AR GLAZŪRAS GABALIŅIEM (30%). Sastāvdaļas: cukurs, kviešu milti, augu eļļas (palmu eļļa, pilnībā hidrogenēta palmu kodolu eļļa, palmu kodolu eļļa), kviešu ciete, kakao pulveris ar samazinātu tauku saturu 4%, olu baltumu un dzeltenumu maisījums, vājpiena pulveris, irdinātāji: amonija hidrogēnkarbonāts, dinātrija difosfāts, nātrija bikarbonāts; sāls, emulgatori: lecitīni (satur soju), aromatizētājs, skābuma regulētājs: pienskābe. Produkts var saturēt zemesriekstus, lazdu riekstus, mandeles un sezama sēklas. Uzglabāt 18±5°C temperatūrā, ja gaisa relatīvais mitrums nepārsniedz 75%. Importētājs ES: UAB „Roshen Nord“, Svajonės g. 27, LT-94101, Klaipėda, LIETUVA. Tel.: +37052409524, e-mail: info@roshen-nord.lt. Izcelsmes valsts: Ukraina. """
,"""(UA) ПЕЧИВО ЗДОБНЕ "LOVITA" CLASSIC COOKIES" З КУСОЧКАМИ ГЛАЗУРІ. Склад: цукор, борошно пшеничне, жир рослинний (негідрогенізована пальмова олія, повністю гідрогенізована пальмоядрова олія, негідрогенізована пальмоядрова олія), крохмаль пшеничний, какао-порошок зі зниженим вмістом жиру 4%, меланж яєчний, молоко сухе знежирене, розпушувачі (гідрокарбонат амонію, дигідропірофосфат натрію, гідрокарбонат натрію), сіль, емульгатори лецитини (містить сою), ароматизатор, регулятор кислотності кислота молочна. Шматочки глазурі какаовмісної – 30%. Може містити арахіс, фундук, мигдаль, кунжут. Зберігати за температури (18±5)°С і відносної вологості повітря не вище 75%. (v181224E) Класичне печиво. Какао. """
,"""(RO)(MD) "LOVITA" CLASSIC COOKIES", BISCUIȚI CU BUCĂȚI DE GLAZURĂ (30%). Ingrediente: zahăr, făină de grâu, grăsimi vegetale (ulei de palmier nehidrogenat, ulei de miez de palmier complet hidrogenat, ulei de miez de palmier nehidrogenat), amidon de grâu, pudră de cacao cu conținut redus de grăsimi 4%, ou lichid pasteurizat, lapte praf degresat, agenți de creștere (carbonat acid de amoniu, difosfat disodic, bicarbonat de sodiu), sare, emulgatori lecitine (conțin soia), arome, regulator de aciditate acid lactic. Poate conține arahide, alune, migdale, semințe de susan. A se păstra la temperatura de (18±5)°С și umiditatea relativă a aerului de 75% maxim. Importator în România: ROSHEN ONE SRL, Blv. Tudor Vladimirescu, nr. 22, Green Gate, unitatea 5, Etaj 5, Sectorul 5, Bucuresti, Romania. Tel. +40761808972. Importator în R. Moldova: SRL «ROSHEN SWEET», R. Moldova, mun. Chișinău, str. I. Zaikin, 10/6, tel.: +37322238301. Ţara de origine: Ucraina. """
,"""Виробник, місцезнаходження: / Изготовитель, место нахождения: / Өндіруші, орналасқан жері: / İstehsalatçı, istehsalat yeri: / Proizvođač, mjesto proizvodnje: / Արտադրող, արտադրման վայրը` """
,"""[ A ] – ПрАТ "Вінницька кондитерська фабрика" (ЧАО "Винницкая кондитерская фабрика"/“Վիննիցկայա հրուշակեղենային ֆաբրիկա” ՄԲԸ/"Винница кондитер фабрикасы" ЖАҚ), вул. Є. Коно- """
,"""вальця, 8, м. Вінниця, 21001, Україна (Украина / Ուկրաինա); адреса виробництва: / адрес производства: / өндіріс мекен-жайы: / արտադրման վայրը` вул. Батозька, 2-К, м. Вінниця, 21015, Україна (Украина / Ուկրաինա). / "Vinnitskaya qənnadı fabriki" ÖSC, E. Konovaltsa küç. 8, Vinnitsa ş., 21001, Ukrayna, istehsalat yeri: Batojskaya küç., 2-K, Vinnitsa ş., 21015, Ukrayna. / PJSC "Vinnytsia Confectionery Factory", st. Ye. Konovaltsia, 8, Vinnytsia, 21001, Ukraine; mjesto proizvodnje: st. Batozka, 2-K, Vinnytsia, 21015, Ukraine. Tel.: 0-800-300-970. www.roshen.com."""
,"""[ H ] – ПрАТ "Київська кондитерська фабрика "Рошен" (ЧАО "Киевская кондитерская фабрика "Рошен"/“Կիևի հրուշակեղենային ֆաբրիկա” “Ռոշեն” ՄԲԸ/ЖАҚ "Киев кондитер фабрикасы "Рошен"), пр-т Науки, 1, м. Київ, 03039, Україна (Украина/Ուկրաինա); адреса виробництва:/адрес производства:/өндіріс мекен-жайы:/արտադրման վայրը` вул. Привокзальна, 82, м. Бориспіль, Київська обл., 08304, Україна (Украина / Ուկրաինա). / "Roşen" Kiev qənnadı fabriki" ÖSC, Nauki pr-ti., 1, Kiev ş., 03039, Ukrayna, istehsalat yeri: Privokzalnaya küç., 82, Borispol ş., Kiev vilayəti, 08304, Ukrayna. /"""
,"""PJSC "Kyiv Roshen Confectionery Factory", Nauky Avenue, 1, Kyiv, 03039, Ukraine.; mjesto proizvodnje: 82 Pryvokzalna St., Boryspil, Kyiv region, 08304, Ukraine. Tel.: 0-800-300-970. www.roshen.com."""
,"""Виробник зазначений літерою після дати виготовлення. / Изготовитель обозначен буквой после даты изготовления. / Өндіруші жасап шығарылған күнінен кейінгі әріппен көрсетілген. / İstehsalçı həriflə istehsal tarixindən sonra göstərilib. / Proizvođač navodi slovo za proizvođača nakon proizvodnje. / Արտադրող, վայրը նշված է համապատասխան  տառով՝ արտադրման տարեթվից հետո: """]




@dataclass
class Match:
    text: str
    len_a: int  # довжина у ref
    len_b: int  # довжина у label
    start_a: int
    start_b: int
    ref_idx: int
    label_idx: int


def all_common_substrings_by_words(ref: str, label: str, min_length_words=2) -> List[Match]:
    """
    Повертає всі спільні підрядки (послідовності слів) довжиною >= min_length_words
    з позиціями у symbovih у ref та label.
    """
    def tokenize_with_positions(text):
        words = []
        positions = []
        for m in re.finditer(r'\w+', text, flags=re.UNICODE):
            words.append(m.group(0))
            positions.append(m.start())
        return words, positions

    ref_words, ref_pos = tokenize_with_positions(ref)
    label_words, label_pos = tokenize_with_positions(label)

    class WordSuffixAutomaton:
        def __init__(self):
            self.next = [{}]
            self.link = [-1]
            self.len = [0]
            self.last = 0

        def extend(self, token):
            p = self.last
            cur = len(self.next)
            self.next.append({})
            self.len.append(self.len[p] + 1)
            self.link.append(0)
            while p >= 0 and token not in self.next[p]:
                self.next[p][token] = cur
                p = self.link[p]
            if p == -1:
                self.link[cur] = 0
            else:
                q = self.next[p][token]
                if self.len[p] + 1 == self.len[q]:
                    self.link[cur] = q
                else:
                    clone = len(self.next)
                    self.next.append(self.next[q].copy())
                    self.len.append(self.len[p] + 1)
                    self.link.append(self.link[q])
                    while p >= 0 and self.next[p][token] == q:
                        self.next[p][token] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            self.last = cur

    sa = WordSuffixAutomaton()
    for w in ref_words:
        sa.extend(w)

    res = []
    n = len(label_words)
    v = 0
    l = 0
    for i in range(n):
        while v and label_words[i] not in sa.next[v]:
            v = sa.link[v]
            l = sa.len[v]
        if label_words[i] in sa.next[v]:
            v = sa.next[v][label_words[i]]
            l += 1
        else:
            v = 0
            l = 0
        if l >= min_length_words:
            pos_ref_word = -1
            vv = v
            ll = l
            while vv:
                if sa.len[sa.link[vv]] < min_length_words:
                    pos_ref_word = sa.len[vv] - l
                    break
                vv = sa.link[vv]
            if pos_ref_word == -1:
                pos_ref_word = i - l + 1
            # --- Перевірка коректності індексів ---
            if (
                0 <= pos_ref_word < len(ref_words) and
                0 <= pos_ref_word + l - 1 < len(ref_words) and
                0 <= i - l + 1 < len(label_words) and
                0 <= i < len(label_words)
            ):
                start_a = ref_pos[pos_ref_word]
                end_a = ref_pos[pos_ref_word + l - 1] + len(ref_words[pos_ref_word + l - 1])
                start_b = label_pos[i - l + 1]
                end_b = label_pos[i] + len(label_words[i])
                text = ref[start_a:end_a]
                res.append(Match(
                    text=text,
                    len_a=end_a - start_a,
                    len_b=end_b - start_b,
                    start_a=start_a,
                    start_b=start_b,
                    ref_idx=ref_idx,
                    label_idx=label_idx
                ))
    return res

def highlight_matches_html(text: str, matches: List[Match], use_a: bool = False) -> str:
    # Сортуємо за початком, щоб вставляти послідовно
    if use_a:
        sorted_matches = sorted(matches, key=lambda m: m.start_a)
    else:
        sorted_matches = sorted(matches, key=lambda m: m.start_b)
    result = []
    pos = 0

    for m in sorted_matches:
        if use_a:
            start = m.start_a
            length = m.len_a
        else:
            start = m.start_b
            length = m.len_b
        if start > pos:
            # Додати незбіг — червоним
            result.append(f'<span style="background-color: #ffcccc;">{text[pos:start]}</span>')
        # Додати збіг — зеленим з ref_idx
        result.append(f'<span style="background-color: #ccffcc;" title="{m.ref_idx}">{text[start:start + length]}</span>')
        pos = start + length

    if pos < len(text):
        result.append(f'<span style="background-color: #ffcccc;">{text[pos:]}</span>')

    return ''.join(result)


def generate_comparison_table_html(label_elements: list[str], ref_elements: list[str], matches: list[Match]) -> str:
    """
    Для кожного label (ліва колонка) будує рядок таблиці.
    У правій колонці — всі ref, для яких у matches є Match з цим label (по ref_idx).
    Для кожного такого ref:
      - Показує ref_idx наприкінці тексту (маленьким шрифтом, білий фон, сіра рамка)
      - Підсвічує тільки ті фрагменти, які співпали з цим label (Match з цим label_idx і ref_idx)
    """
    from collections import defaultdict
    # Групуємо Match по (label_idx, ref_idx)
    matches_by_label_ref = defaultdict(list)
    for m in matches:
        matches_by_label_ref[(m.label_idx, m.ref_idx)].append(m)

    html_rows = []
    for label_idx, label in enumerate(label_elements):
        # Знаходимо всі ref_idx, які мають Match з цим label
        # Сортуємо ref_idx за мінімальним start_a серед matches для цього label_idx/ref_idx
        ref_idxs = list({m.ref_idx for m in matches if m.label_idx == label_idx})
        ref_idxs.sort(key=lambda ref_idx: min((m.start_b for m in matches_by_label_ref[(label_idx, ref_idx)]), default=1e9))
        # Ліва колонка — label
        left_html = f"<div class='label-cell'><pre>{highlight_matches_html(label, [m for m in matches if m.label_idx == label_idx])}</pre></div>"
        # Права колонка — всі ref з Match
        right_blocks = []
        for ref_idx in ref_idxs:
            ref = ref_elements[ref_idx]
            ref_matches = matches_by_label_ref[(label_idx, ref_idx)]
            ref_html = highlight_matches_html(ref, ref_matches, use_a=True)
            # ref_idx у кінці тексту, маленьким шрифтом, білий фон, сіра рамка, margin-left
            ref_html_with_idx = f"{ref_html}<span style='font-size:0.8em; background:#fff; border:1px solid #ccc; color:#888; margin-left:1em; padding:0 0.3em; border-radius:3px;'>{ref_idx}</span>"
            right_blocks.append(f"<div class='ref-block'><pre>{ref_html_with_idx}</pre></div>")
        right_html = "<div class='refs-cell'>" + "<hr>".join(right_blocks) + "</div>"
        html_rows.append(f"<tr><td class='label-col'>{left_html}</td><td class='refs-col'>{right_html}</td></tr>")
    style = """
    <style>
    table.comparison-table { width: 100%; border-collapse: collapse; }
    .label-col, .refs-col { vertical-align: top; border: 1px solid #ccc; padding: 0.5em; }
    .label-cell pre, .ref-block pre { white-space: pre-wrap; word-break: break-all; font-family: 'Consolas', 'Menlo', 'Monaco', monospace; }
    .ref-block { margin-bottom: 1em; }
    .ref-idx { font-size: 0.9em; color: #555; margin-bottom: 0.2em; }
    </style>
    """
    table_html = "<table class='comparison-table'>" + "".join(html_rows) + "</table>"
    return f"<html><head>{style}</head><body>{table_html}</body></html>"



def remove_overlapping_by_label(matches: list) -> list:
    # Групуємо по label_idx
    from collections import defaultdict
    by_label = defaultdict(list)
    for m in matches:
        by_label[m.label_idx].append(m)
    survivors = []
    for label_idx, group in by_label.items():
        # Сортуємо за спаданням довжини, потім за start_b
        group = sorted(group, key=lambda m: (-m.len_b, m.start_b))
        occupied = []
        for m in group:
            m_start = m.start_b
            m_end = m.start_b + m.len_b
            overlap = False
            for s, e in occupied:
                if not (m_end <= s or m_start >= e):
                    overlap = True
                    break
            if not overlap:
                survivors.append(m)
                occupied.append((m_start, m_end))
    return survivors

def remove_overlapping_by_ref(matches: list) -> list:
    # Групуємо по label_idx
    from collections import defaultdict
    by_ref = defaultdict(list)
    for m in matches:
        by_ref[m.ref_idx].append(m)
    survivors = []
    for куа_idx, group in by_ref.items():
        # Сортуємо за спаданням довжини, потім за start_b
        group = sorted(group, key=lambda m: (-m.len_a, m.start_a))
        occupied = []
        for m in group:
            m_start = m.start_a
            m_end = m.start_a + m.len_a
            overlap = False
            for s, e in occupied:
                if not (m_end <= s or m_start >= e):
                    overlap = True
                    break
            if not overlap:
                survivors.append(m)
                occupied.append((m_start, m_end))
    return survivors


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    def get_text_no_extra_space(el):
        # Якщо це просто текст
        if isinstance(el, NavigableString):
            return str(el)
        # Якщо це тег
        result = ""
        for child in el.children:
            result += get_text_no_extra_space(child)
        return result
    # Збираємо текст без зайвих пробілів
    text = get_text_no_extra_space(soup)
    # Додатково прибираємо зайві пробіли на краях і між словами
    text = ' '.join(text.split())
    return text

def normalize_text(text: str) -> str:
    text = text.replace(' ', ' ')
    text = text.replace('–', '-')
    # Прибрати пробіли перед комами, крапками, двокрапками, крапками з комою
    text = re.sub(r'\s+([,.;:()])', r'\1', text)
    # Замінити кілька пробілів на один
    text = re.sub(r'\s+', ' ', text)
    #. Прибрати пробіли на початку і в кінці
    text = text.strip()
    
    return text


ref_elements = [clean_html(ref) for ref in ref_elements]
ref_elements = [normalize_text(ref) for ref in ref_elements]
label_elements = [normalize_text(label) for label in label_elements]

results = []


all_matches = []
for label_idx, label in enumerate(label_elements):
    for ref_idx, ref in enumerate(ref_elements):
        #matches = all_common_substrings_with_positions(ref, label, min_length=10)
        matches = all_common_substrings_by_words(ref, label, min_length_words=2)
        
        # Додаємо label_idx до кожного Match
        for m in matches:
            m.label_idx = label_idx
            m.ref_idx = ref_idx
            all_matches.append(m)

all_matches = remove_overlapping_by_label(all_matches)
all_matches = remove_overlapping_by_ref(all_matches)
results.extend(all_matches)


html = generate_comparison_table_html(label_elements, ref_elements, results)
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html)


print("**** Check overlapping *****************")
for i, m1 in enumerate(results):
       for j, m2 in enumerate(results):
           if i >= j: continue
           if m1.ref_idx == m2.ref_idx:
               a1, b1 = m1.start_a, m1.start_a + m1.len_a
               a2, b2 = m2.start_a, m2.start_a + m2.len_a
               if not (b1 <= a2 or b2 <= a1):
                   print(f"!!! overlapping !!!")
                   print(f"idx {i} і {j} по ref_idx={m1.ref_idx}: [{a1},{b1}) & [{a2},{b2})")
