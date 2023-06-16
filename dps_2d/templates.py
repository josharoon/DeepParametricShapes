import numpy as np
import torch

from pyGutils.viz import plotQuadraticSpline

n_loops = {
    'A': 2, 'B': 3, 'C': 1, 'D': 2, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1,
    'O': 2, 'P': 2, 'Q': 2, 'R': 2, 'S': 1, 'T': 1, 'U': 1, 'V': 1, 'W': 1, 'X': 1, 'Y': 1, 'Z': 1
}
n_loops_eye={'P':1,'I':1,'C':2}

#topology = [15, 4, 4]
topology = [8, 4, 4]
topology2 = [8, 4, 4]

eye_templates=[[0.5071626901626587, 0.1450844556093216, 0.5451360940933228, 0.1536296010017395, 0.5746654272079468, 0.17936621606349945, 0.6041947603225708, 0.2051028311252594, 0.6252800226211548,
                0.24803093075752258, 0.6463652849197388, 0.29095903038978577, 0.6590064764022827, 0.351078599691391, 0.6716477274894714, 0.4111981689929962, 0.6758449077606201, 0.48281246423721313,
                0.6716477274894714, 0.5544267892837524, 0.6590064764022827, 0.6145463585853577, 0.6463652849197388, 0.6746659278869629, 0.6252800226211548, 0.7175940275192261, 0.6041947603225708,
                0.7605221271514893, 0.5746654272079468, 0.7862586975097656, 0.5451360940933228, 0.8119953274726868, 0.5099608302116394, 0.8205404877662659, 0.47478556632995605, 0.8119953274726868, 0.4452562630176544,
                0.7862586975097656, 0.4157269299030304, 0.7605221271514893, 0.3946416676044464, 0.7175940275192261, 0.3735564053058624, 0.6746659278869629, 0.3609151840209961, 0.6145463585853577,
                0.34827396273612976, 0.5544267892837524, 0.3440767824649811, 0.48281246423721313, 0.34827396273612976, 0.4111981689929962, 0.3609151840209961, 0.351078599691391, 0.3735564053058624,
                0.29095903038978577, 0.3946416676044464, 0.24803093075752258, 0.4368121922016144, 0.16217473149299622, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],[0.4620819687843323, 0.9512184262275696, 0.4873883128166199, 0.9837473630905151, 0.510094404220581, 0.9503145217895508, 0.5162566900253296, 0.9357560276985168, 0.5248708724975586,
                0.8900860548019409, 0.5250937938690186, 0.8373408317565918, 0.52614825963974, 0.7631486654281616, 0.5272027850151062, 0.6889565587043762, 0.5281574726104736, 0.6198570728302002,
                0.5300669074058533, 0.48165807127952576, 0.5315771102905273, 0.3638297915458679, 0.5330873727798462, 0.2460014969110489, 0.5341984033584595, 0.14854387938976288, 0.5353094935417175,
                0.051086269319057465, 0.540142297744751, -0.00030781328678131104, 0.49196428060531616, 0.025418564677238464, 0.44394105672836304, -0.0025787455961108208, 0.4484642744064331,
                0.0534745492041111, 0.4489050507545471, 0.1564752161502838, 0.44934582710266113, 0.2594758868217468, 0.4499799609184265, 0.38139307498931885, 0.4502970576286316, 0.44235166907310486, 0.4506624639034271,
                0.5080394148826599, 0.45102787017822266, 0.5737271308898926, 0.45144161581993103, 0.6441439986228943, 0.4518553614616394, 0.714560866355896, 0.4523174464702606, 0.789706826210022, 0.45277953147888184,
                0.8648528456687927, 0.4526141881942749, 0.9162777662277222, 0.4595615863800049, 0.9421639442443848, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],[0.4620819687843323, 0.9512184262275696, 0.4873883128166199, 0.9837473630905151, 0.510094404220581, 0.9503145217895508, 0.5162566900253296, 0.9357560276985168, 0.5248708724975586, 0.8900860548019409, 0.5250937938690186, 0.8373408317565918, 0.52614825963974, 0.7631486654281616, 0.5272027850151062, 0.6889565587043762, 0.5281574726104736, 0.6198570728302002, 0.5300669074058533, 0.48165807127952576, 0.5315771102905273, 0.3638297915458679, 0.5330873727798462, 0.2460014969110489, 0.5341984033584595, 0.14854387938976288, 0.5353094935417175, 0.051086269319057465, 0.540142297744751, -0.00030781328678131104, 0.49196428060531616, 0.025418564677238464, 0.44394105672836304, -0.0025787455961108208, 0.4484642744064331, 0.0534745492041111, 0.4489050507545471, 0.1564752161502838, 0.44934582710266113, 0.2594758868217468, 0.4499799609184265, 0.38139307498931885, 0.4502970576286316, 0.44235166907310486, 0.4506624639034271, 0.5080394148826599, 0.45102787017822266, 0.5737271308898926, 0.45144161581993103, 0.6441439986228943, 0.4518553614616394, 0.714560866355896, 0.4523174464702606, 0.789706826210022, 0.45277953147888184, 0.8648528456687927, 0.4526141881942749, 0.9162777662277222, 0.4595615863800049, 0.9421639442443848, 0.5071626901626587, 0.1450844556093216, 0.6590561866760254, 0.17926500737667084, 0.6758449077606201, 0.48281246423721313, 0.6590561866760254, 0.7863599061965942, 0.5099608302116394, 0.8205404877662659, 0.3608654737472534, 0.7863599061965942, 0.3440767824649811, 0.48281246423721313, 0.3608654737472534, 0.17926500737667084, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],

               [0.35340243577957153, 0.6183240413665771, 0.5077964067459106, 0.644997775554657, 0.6463263630867004, 0.6175828576087952, 0.6839224100112915, 0.6056448817253113, 0.7364773750305176, 0.5681955814361572, 0.7448798418045044, 0.4599308967590332, 0.7565291523933411, 0.3466077446937561, 0.7681785225868225, 0.233284592628479, 0.7773924469947815, 0.13666538894176483, 0.7866063714027405, 0.04004618898034096, 0.7933849096298218, -0.039869051426649094, 0.8001634478569031, -0.11978428810834885, 0.8296486139297485, -0.16192743182182312, 0.775021493434906, -0.1408524215221405, 0.6952524781227112, -0.14084555208683014, 0.6154834032058716, -0.14083868265151978, 0.5357143878936768, -0.1408318132162094, 0.45594534277915955, -0.14082494378089905, 0.37617629766464233, -0.14081807434558868, 0.2964072823524475, -0.14081120491027832, 0.2427247166633606, -0.16378962993621826, 0.2703208029270172, -0.11782591044902802, 0.2730100452899933, -0.03336536884307861, 0.27569931745529175, 0.05109517648816109, 0.27956822514533997, 0.15106728672981262, 0.2834371328353882, 0.25103938579559326, 0.28848570585250854, 0.36652302742004395, 0.2935342788696289, 0.482006698846817, 0.2956395745277405, 0.5896728038787842, 0.3380257487297058, 0.6108993887901306, 0.5071626901626587, 0.1450844556093216, 0.6590561866760254, 0.17926500737667084, 0.6758449077606201, 0.48281246423721313, 0.6590561866760254, 0.7863599061965942, 0.5099608302116394, 0.8205404877662659, 0.3608654737472534, 0.7863599061965942, 0.3440767824649811, 0.48281246423721313, 0.3608654737472534, 0.17926500737667084, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.3369961380958557, 0.24964231252670288, 0.4913901090621948, 0.2190651148557663, 0.6299200654029846, 0.25049203634262085, 0.6675161123275757, 0.26417696475982666, 0.7200710773468018, 0.30710670351982117, 0.7284735441207886, 0.43121495842933655, 0.7401228547096252, 0.5611220002174377, 0.7517722249031067, 0.6910290718078613, 0.7609861493110657, 0.8017876744270325, 0.7702000737190247, 0.9125462770462036, 0.776978611946106, 1.004156470298767, 0.7837571501731873, 1.095766544342041, 0.8132423162460327, 1.1440770626068115, 0.7586151957511902, 1.119917869567871, 0.6788461804389954, 1.1199100017547607, 0.5990771055221558, 1.1199021339416504, 0.5193080902099609, 1.11989426612854, 0.4395390450954437, 1.1198862791061401, 0.3597699999809265, 1.1198784112930298, 0.2800009846687317, 1.1198705434799194, 0.22631841897964478, 1.1462116241455078, 0.2539145052433014, 1.0935215950012207, 0.25660374760627747, 0.9967009425163269, 0.2592930197715759, 0.8998802900314331, 0.26316192746162415, 0.7852781414985657, 0.26703083515167236, 0.6706759929656982, 0.2720794081687927, 0.5382922887802124, 0.2771279811859131, 0.4059085547924042, 0.27923327684402466, 0.2824864685535431, 0.32161945104599, 0.25815349817276, 0.5071626901626587, 0.1450844556093216, 0.6590561866760254, 0.17926500737667084, 0.6758449077606201, 0.48281246423721313, 0.6590561866760254, 0.7863599061965942, 0.5099608302116394, 0.8205404877662659, 0.3608654737472534, 0.7863599061965942, 0.3440767824649811, 0.48281246423721313, 0.3608654737472534, 0.17926500737667084, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.3369961380958557, 0.24964231252670288, 0.4913901090621948, 0.2190651148557663, 0.6299200654029846, 0.25049203634262085, 0.6675161123275757, 0.26417696475982666, 0.7200710773468018, 0.30710670351982117, 0.7284735441207886, 0.43121495842933655, 0.7401228547096252, 0.5611220002174377, 0.7517722249031067, 0.6910290718078613, 0.7609861493110657, 0.8017876744270325, 0.7702000737190247, 0.9125462770462036, 0.776978611946106, 1.004156470298767, 0.7837571501731873, 1.095766544342041, 0.8132423162460327, 1.1440770626068115, 0.7586151957511902, 1.119917869567871, 0.6788461804389954, 1.1199100017547607, 0.5990771055221558, 1.1199021339416504, 0.5193080902099609, 1.11989426612854, 0.4395390450954437, 1.1198862791061401, 0.3597699999809265, 1.1198784112930298, 0.2800009846687317, 1.1198705434799194, 0.22631841897964478, 1.1462116241455078, 0.2539145052433014, 1.0935215950012207, 0.25660374760627747, 0.9967009425163269, 0.2592930197715759, 0.8998802900314331, 0.26316192746162415, 0.7852781414985657, 0.26703083515167236, 0.6706759929656982, 0.2720794081687927, 0.5382922887802124, 0.2771279811859131, 0.4059085547924042, 0.27923327684402466, 0.2824864685535431, 0.32161945104599, 0.25815349817276, 0.4921056926250458, 0.013264830224215984, 0.9941760897636414, -0.016413478180766106, 0.9890291690826416, 0.47768181562423706, 0.9327161312103271, 0.9770945310592651, 0.48080357909202576, 0.9771783351898193, 0.041278544813394547, 0.9979960322380066, 0.007220442406833172, 0.49848997592926025, -0.0009838985279202461, 0.007729066535830498, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.3369961380958557, 0.24964231252670288, 0.4913901090621948, 0.2190651148557663, 0.6299200654029846, 0.25049203634262085, 0.6675161123275757, 0.26417696475982666, 0.7200710773468018, 0.30710670351982117, 0.7284735441207886, 0.43121495842933655, 0.7401228547096252, 0.5611220002174377, 0.7517722249031067, 0.6910290718078613, 0.7609861493110657, 0.8017876744270325, 0.7702000737190247, 0.9125462770462036, 0.776978611946106, 1.004156470298767, 0.7837571501731873, 1.095766544342041, 0.8132423162460327, 1.1440770626068115, 0.7586151957511902, 1.119917869567871, 0.6788461804389954, 1.1199100017547607, 0.5990771055221558, 1.1199021339416504, 0.5193080902099609, 1.11989426612854, 0.4395390450954437, 1.1198862791061401, 0.3597699999809265, 1.1198784112930298, 0.2800009846687317, 1.1198705434799194, 0.22631841897964478, 1.1462116241455078, 0.2539145052433014, 1.0935215950012207, 0.25660374760627747, 0.9967009425163269, 0.2592930197715759, 0.8998802900314331, 0.26316192746162415, 0.7852781414985657, 0.26703083515167236, 0.6706759929656982, 0.2720794081687927, 0.5382922887802124, 0.2771279811859131, 0.4059085547924042, 0.27923327684402466, 0.2824864685535431, 0.32161945104599, 0.25815349817276, 0.5223376750946045, 0.04156792163848877, 0.8120323419570923, 0.02604616992175579, 0.8090624809265137, 0.2844579815864563, 0.7765698432922363, 0.5456507802009583, 0.5158163905143738, 0.5456946492195129, 0.2622103989124298, 0.5565822720527649, 0.24255888164043427, 0.2953406572341919, 0.23782499134540558, 0.03867271915078163, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.35340240597724915, 0.85055011510849, 0.4305993914604187, 0.8401873707771301, 0.507771909236908, 0.8404778838157654, 0.5849443674087524, 0.8407683968544006, 0.6463263630867004, 0.8511260747909546, 0.6839224100112915, 0.8604018092155457, 0.7364773750305176, 0.8894999027252197, 0.7448798418045044, 0.9736215472221375, 0.7565291523933411, 1.061673641204834, 0.7681785225868225, 1.1497256755828857, 0.7773924469947815, 1.2247987985610962, 0.7958203554153442, 1.374945044517517, 0.8296486139297485, 1.4568054676055908, 0.775021493434906, 1.4404301643371582, 0.6952524781227112, 1.4404247999191284, 0.6154834032058716, 1.4404195547103882, 0.5357143878936768, 1.4404141902923584, 0.45594534277915955, 1.4404088258743286, 0.37617629766464233, 1.4404035806655884, 0.2964072525501251, 1.4403982162475586, 0.2427246868610382, 1.4582524299621582, 0.2703207731246948, 1.4225386381149292, 0.2730100154876709, 1.3569127321243286, 0.27569928765296936, 1.2912869453430176, 0.2795681953430176, 1.2136086225509644, 0.2834371030330658, 1.1359302997589111, 0.28848567605018616, 1.0461994409561157, 0.2935342490673065, 0.9564687013626099, 0.2956395745277405, 0.8728121519088745, 0.3380257189273834, 0.8563190698623657, 0.5292627215385437, 0.04996154084801674, 0.7902775406837463, 0.03577466681599617, 0.7876017093658447, 0.2719630002975464, 0.7583259344100952, 0.5106932520866394, 0.5233870148658752, 0.5107333064079285, 0.29488804936408997, 0.5206846594810486, 0.27718204259872437, 0.2819097936153412, 0.2729167938232422, 0.0473153293132782, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.3393399119377136, 0.022404801100492477, 0.49373388290405273, -0.02014458179473877, 0.6322638392448425, 0.02358723059296608, 0.6698598861694336, 0.04263034462928772, 0.7224148511886597, 0.1023687869310379, 0.7237749099731445, 0.17136205732822418, 0.7302084565162659, 0.26840898394584656, 0.7366419434547424, 0.36545586585998535, 0.7424666285514832, 0.4558413326740265, 0.7541159987449646, 0.6366122364997864, 0.7633299231529236, 0.7907373309135437, 0.7725438475608826, 0.944862425327301, 0.7793223857879639, 1.0723416805267334, 0.7861009240150452, 1.1998209953308105, 0.8155860900878906, 1.2670469284057617, 0.6811899542808533, 1.2334176301956177, 0.5216518640518188, 1.2333956956863403, 0.3621137738227844, 1.233373761177063, 0.22866219282150269, 1.2700176239013672, 0.2562582790851593, 1.1966971158981323, 0.2589475214481354, 1.0619672536849976, 0.26163679361343384, 0.9272374510765076, 0.26550570130348206, 0.7677638530731201, 0.2693746089935303, 0.6082902550697327, 0.27442318201065063, 0.4240729808807373, 0.2769474685192108, 0.33196431398391724, 0.27976667881011963, 0.2336697280406952, 0.28258588910102844, 0.13537514209747314, 0.28157705068588257, 0.06810883432626724, 0.3239632248878479, 0.03424850106239319, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.4669370949268341, 0.6683377623558044, 0.5132730007171631, 0.6589604020118713, 0.5548478364944458, 0.6685983538627625, 0.5661311149597168, 0.6727952361106873, 0.5819035768508911, 0.6859608888626099, 0.5886523127555847, 0.7697340846061707, 0.5941827893257141, 0.8376684188842773, 0.5997132658958435, 0.9056028127670288, 0.6098656058311462, 0.9426409602165222, 0.5216516256332397, 0.9352246522903442, 0.43372106552124023, 0.9432955980300903, 0.44245609641075134, 0.9028971791267395, 0.44477832317352295, 0.8326053619384766, 0.44710057973861694, 0.7623134851455688, 0.44960159063339233, 0.6784103512763977, 0.46232229471206665, 0.6709479689598083, 0.5292627215385437, 0.04996154084801674, 0.7902775406837463, 0.03577466681599617, 0.7876017093658447, 0.2719630002975464, 0.7583259344100952, 0.5106932520866394, 0.5233870148658752, 0.5107333064079285, 0.29488804936408997, 0.5206846594810486, 0.27718204259872437, 0.2819097936153412, 0.2729167938232422, 0.0473153293132782, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

eye_templates2=[[0.3393399119377136, 0.022404801100492477, 0.49373388290405273, -0.02014458179473877, 0.6322638392448425, 0.02358723059296608, 0.6698598861694336, 0.04263034462928772, 0.7224148511886597, 0.1023687869310379, 0.7449020147323608, 0.48248714208602905, 0.7633299231529236, 0.7907373309135437, 0.7817578315734863, 1.0989875793457031, 0.8155860900878906, 1.2670469284057617, 0.5216518640518188, 1.2333956956863403, 0.22866219282150269, 1.2700176239013672, 0.25776785612106323, 1.086711049079895, 0.26550570130348206, 0.7677638530731201, 0.2732435464859009, 0.4488166868686676, 0.28157705068588257, 0.06810883432626724, 0.3239632248878479, 0.03424850106239319, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.35574620962142944, 0.1536547988653183, 0.5101401805877686, 0.11110541224479675, 0.6381590366363525, 0.13786719739437103, 0.791377067565918, 0.34358057379722595, 0.9473371505737305, 0.607945442199707, 1.1398439407348633, 0.9380753040313721, 1.402619481086731, 1.1068389415740967, 1.2282634973526, 1.0909677743911743, 0.9918147921562195, 1.0901364088058472, 0.7553660869598389, 1.0893051624298096, 0.518917441368103, 1.0884737968444824, 0.04602007940411568, 1.0868113040924072, -0.36305639147758484, 1.0922831296920776, -0.11929136514663696, 1.0171293020248413, 0.048334069550037384, 0.562210202217102, 0.21458788216114044, 0.32409214973449707, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5292627215385437, 0.04996154084801674, 0.659770131111145, 0.042868103832006454, 0.7236844301223755, 0.09804993867874146, 0.7875986695289612, 0.15323176980018616, 0.7876017093658447, 0.2719630002975464, 0.7743048071861267, 0.39196521043777466, 0.706490159034729, 0.4512801468372345, 0.6386755108833313, 0.5105950832366943, 0.5233870148658752, 0.5107333064079285, 0.41131851077079773, 0.5158271789550781, 0.3480004668235779, 0.4587966799736023, 0.2846824526786804, 0.40176618099212646, 0.27718204259872437, 0.2819097936153412, 0.2764020264148712, 0.1641436070203781, 0.3382332921028137, 0.10634109377861023, 0.4000645577907562, 0.04853859171271324, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                ,[0.35574620962142944, 0.1536547988653183, 0.5101401805877686, 0.11110541224479675, 0.6381590366363525, 0.13786719739437103, 0.791377067565918, 0.34358057379722595, 0.9473371505737305, 0.607945442199707, 1.1398439407348633, 0.9380753040313721, 1.402619481086731, 1.1068389415740967, 1.2282634973526, 1.0909677743911743, 0.9918147921562195, 1.0901364088058472, 0.7553660869598389, 1.0893051624298096, 0.518917441368103, 1.0884737968444824, 0.04602007940411568, 1.0868113040924072, -0.36305639147758484, 1.0922831296920776, -0.11929136514663696, 1.0171293020248413, 0.048334069550037384, 0.562210202217102, 0.21458788216114044, 0.32409214973449707, 0.5292627215385437, 0.04996154084801674, 0.7902775406837463, 0.03577466681599617, 0.7876017093658447, 0.2719630002975464, 0.7583259344100952, 0.5106932520866394, 0.5233870148658752, 0.5107333064079285, 0.29488804936408997, 0.5206846594810486, 0.27718204259872437, 0.2819097936153412, 0.2729167938232422, 0.0473153293132782, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.4839566648006439, 0.3014914393424988, 0.5092630386352539, 0.2800461947917938, 0.5319691300392151, 0.30208727717399597, 0.5381314158439636, 0.311685174703598, 0.5467455387115479, 0.34179383516311646, 0.5504313707351685, 0.5333754420280457, 0.5534518957138062, 0.6887351274490356, 0.5564723610877991, 0.8440948128700256, 0.5620170831680298, 0.9287976026535034, 0.5138390064239502, 0.911837100982666, 0.4658157527446747, 0.9302946329116821, 0.4705863893032074, 0.8379071950912476, 0.47185468673706055, 0.6771562099456787, 0.4731229543685913, 0.5164052844047546, 0.47448885440826416, 0.3245265483856201, 0.48143628239631653, 0.3074606657028198, 0.5292627215385437, 0.04996154084801674, 0.7902775406837463, 0.03577466681599617, 0.7876017093658447, 0.2719630002975464, 0.7583259344100952, 0.5106932520866394, 0.5233870148658752, 0.5107333064079285, 0.29488804936408997, 0.5206846594810486, 0.27718204259872437, 0.2819097936153412, 0.2729167938232422, 0.0473153293132782, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.4669370949268341, 0.6683377623558044, 0.5132730007171631, 0.6589604020118713, 0.5548478364944458, 0.6685983538627625, 0.5661311149597168, 0.6727952361106873, 0.5819035768508911, 0.6859608888626099, 0.5886523127555847, 0.7697340846061707, 0.5941827893257141, 0.8376684188842773, 0.5997132658958435, 0.9056028127670288, 0.6098656058311462, 0.9426409602165222, 0.5216516256332397, 0.9352246522903442, 0.43372106552124023, 0.9432955980300903, 0.44245609641075134, 0.9028971791267395, 0.44477832317352295, 0.8326053619384766, 0.44710057973861694, 0.7623134851455688, 0.44960159063339233, 0.6784103512763977, 0.46232229471206665, 0.6709479689598083, 0.5292627215385437, 0.04996154084801674, 0.7902775406837463, 0.03577466681599617, 0.7876017093658447, 0.2719630002975464, 0.7583259344100952, 0.5106932520866394, 0.5233870148658752, 0.5107333064079285, 0.29488804936408997, 0.5206846594810486, 0.27718204259872437, 0.2819097936153412, 0.2729167938232422, 0.0473153293132782, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

simple_templates = [
    [0.4*x + 0.5 for y in np.linspace(0, 2*np.pi, 30) for x in (np.cos(y), np.sin(y))] + [0.5]*32,
    [0.4*x + 0.5 for y in np.linspace(0, 2*np.pi, 30) for x in (np.cos(y), np.sin(y))] + \
        [0.2*x + 0.5 for y in np.linspace(0, 2*np.pi, 8) for x in (np.cos(y), np.sin(y))] + [0.5]*16,
    [0.4*x + 0.5 for y in np.linspace(0, 2*np.pi, 30) for x in (np.cos(y), np.sin(y))] + \
        [0.15*x + 0.5 for y in np.linspace(0, 2*np.pi, 8) for x in (np.cos(y), np.sin(y)+1.1)] + \
        [0.15*x + 0.5 for y in np.linspace(0, 2*np.pi, 8) for x in (np.cos(y), np.sin(y)-1.1)]
]

letter_templates = [
    [0.17, 0.9, 0.23, 0.9, 0.28, 0.9, 0.32, 0.77, 0.36, 0.65, 0.42, 0.65, 0.5, 0.65, 0.58, 0.65, 0.65,
        0.65, 0.68, 0.76, 0.73, 0.9, 0.78, 0.9, 0.84, 0.9, 0.81, 0.79, 0.76, 0.67, 0.74, 0.59, 0.7, 0.48, 0.66,
        0.36, 0.63, 0.27, 0.6, 0.2, 0.57, 0.1, 0.52, 0.1, 0.44, 0.1, 0.42, 0.17, 0.39, 0.27, 0.36, 0.34, 0.33,
        0.43, 0.3, 0.52, 0.27, 0.6, 0.24, 0.71, 0.48, 0.29, 0.43, 0.42, 0.38, 0.56, 0.5, 0.57, 0.62, 0.57, 0.58,
        0.45, 0.54, 0.32, 0.5, 0.19] + [0.5]*16,
    [0.25, 0.12, 0.25, 0.21, 0.25, 0.27, 0.25, 0.34, 0.25, 0.41, 0.25, 0.48, 0.25, 0.55, 0.25, 0.62, 0.25,
        0.68, 0.25, 0.75, 0.25, 0.89, 0.34, 0.9, 0.43, 0.9, 0.52, 0.89, 0.62, 0.87, 0.69, 0.82, 0.73, 0.75,
        0.75, 0.68, 0.73, 0.59, 0.67, 0.52, 0.58, 0.47, 0.66, 0.43, 0.7, 0.36, 0.72, 0.31, 0.7, 0.23, 0.66,
        0.17, 0.59, 0.13, 0.53, 0.11, 0.46, 0.1, 0.37, 0.11, 0.35, 0.19, 0.35, 0.31, 0.35, 0.44, 0.5, 0.44, 0.6,
        0.38, 0.62, 0.31, 0.56, 0.21, 0.48, 0.18, 0.35, 0.52, 0.35, 0.65, 0.35, 0.82, 0.47, 0.82, 0.6, 0.77,
        0.64, 0.68, 0.6, 0.57, 0.51, 0.53],
    [0.39, 0.16, 0.27, 0.25, 0.21, 0.41, 0.19, 0.52, 0.22, 0.63, 0.25, 0.76, 0.33, 0.84, 0.43, 0.87, 0.56,
        0.9, 0.68, 0.89, 0.8, 0.86, 0.79, 0.82, 0.78, 0.79, 0.64, 0.82, 0.52, 0.8, 0.43, 0.76, 0.36, 0.68, 0.32,
        0.57, 0.32, 0.47, 0.34, 0.34, 0.42, 0.26, 0.48, 0.22, 0.57, 0.18, 0.7, 0.19, 0.78, 0.22, 0.79, 0.18,
        0.81, 0.13, 0.73, 0.11, 0.62, 0.1, 0.51, 0.1] + [0.5 ]*32,
    [0.18, 0.12, 0.18, 0.2, 0.18, 0.26, 0.18, 0.33, 0.18, 0.37, 0.18, 0.45, 0.18, 0.52, 0.18, 0.58, 0.18,
        0.65, 0.18, 0.75, 0.18, 0.89, 0.27, 0.9, 0.34, 0.9, 0.44, 0.9, 0.52, 0.88, 0.61, 0.85, 0.67, 0.82, 0.74,
        0.76, 0.78, 0.69, 0.81, 0.6, 0.82, 0.52, 0.82, 0.42, 0.79, 0.31, 0.73, 0.22, 0.66, 0.17, 0.6, 0.12,
        0.54, 0.11, 0.47, 0.1, 0.38, 0.1, 0.3, 0.1, 0.28, 0.19, 0.28, 0.45, 0.28, 0.81, 0.48, 0.81, 0.67, 0.68,
        0.71, 0.5, 0.68, 0.32, 0.58, 0.21] + [0.5]*16,
    [0.28, 0.1, 0.28, 0.19, 0.28, 0.27, 0.28, 0.34, 0.28, 0.44, 0.28, 0.52, 0.28, 0.61, 0.28, 0.71, 0.28,
        0.9, 0.5, 0.9, 0.72, 0.9, 0.72, 0.85, 0.72, 0.81, 0.56, 0.81, 0.38, 0.81, 0.38, 0.69, 0.38, 0.52, 0.52,
        0.52, 0.69, 0.52, 0.69, 0.5, 0.69, 0.44, 0.54, 0.44, 0.38, 0.44, 0.38, 0.32, 0.38, 0.19, 0.54, 0.19,
        0.71, 0.19, 0.71, 0.15, 0.71, 0.1, 0.56, 0.1] + [0.5]*32,
    [0.28, 0.1, 0.28, 0.19, 0.28, 0.27, 0.28, 0.34, 0.28, 0.44, 0.28, 0.52, 0.28, 0.61, 0.28, 0.71, 0.28,
        0.9, 0.33, 0.9, 0.39, 0.9, 0.39, 0.85, 0.39, 0.81, 0.39, 0.77, 0.39, 0.68, 0.39, 0.6, 0.39, 0.52, 0.53,
        0.52, 0.7, 0.52, 0.7, 0.5, 0.7, 0.44, 0.55, 0.44, 0.39, 0.44, 0.39, 0.32, 0.39, 0.19, 0.55, 0.19, 0.72,
        0.19, 0.72, 0.15, 0.72, 0.1, 0.57, 0.1] + [0.5]*32,
    [0.8, 0.14, 0.64, 0.1, 0.46, 0.11, 0.34, 0.17, 0.23, 0.27, 0.19, 0.39, 0.17, 0.53, 0.2, 0.68, 0.33,
        0.84, 0.49, 0.9, 0.83, 0.86, 0.83, 0.68, 0.83, 0.48, 0.72, 0.48, 0.57, 0.48, 0.57, 0.52, 0.57, 0.56,
        0.62, 0.56, 0.73, 0.56, 0.73, 0.65, 0.73, 0.8, 0.6, 0.82, 0.42, 0.77, 0.3, 0.57, 0.31, 0.38, 0.4, 0.26,
        0.48, 0.21, 0.6, 0.2, 0.78, 0.22, 0.79, 0.18] + [0.5]*32,
    [0.2, 0.1, 0.2, 0.23, 0.2, 0.42, 0.2, 0.65, 0.2, 0.9, 0.24, 0.9, 0.31, 0.9, 0.31, 0.74, 0.31, 0.52,
        0.45, 0.52, 0.69, 0.52, 0.69, 0.64, 0.69, 0.9, 0.74, 0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.44, 0.8, 0.31, 0.8,
        0.1, 0.75, 0.1, 0.69, 0.1, 0.69, 0.21, 0.69, 0.43, 0.61, 0.43, 0.5, 0.43, 0.4, 0.43, 0.31, 0.43, 0.31,
        0.27, 0.31, 0.1, 0.25, 0.1] + [0.5]*32,
    [0.45, 0.1, 0.45, 0.15, 0.45, 0.21, 0.45, 0.29, 0.45, 0.34, 0.45, 0.4, 0.45, 0.43, 0.45, 0.5, 0.45,
        0.55, 0.45, 0.61, 0.45, 0.65, 0.45, 0.7, 0.45, 0.74, 0.45, 0.8, 0.45, 0.9, 0.49, 0.9, 0.55, 0.9, 0.55,
        0.86, 0.55, 0.75, 0.55, 0.68, 0.55, 0.62, 0.55, 0.57, 0.55, 0.51, 0.55, 0.44, 0.55, 0.38, 0.55, 0.32,
        0.55, 0.25, 0.55, 0.19, 0.55, 0.1, 0.5, 0.1] + [0.5]*32,
    [0.55, 0.1, 0.55, 0.16, 0.55, 0.22, 0.55, 0.28, 0.55, 0.33, 0.55, 0.41, 0.55, 0.46, 0.55, 0.54, 0.55,
        0.62, 0.55, 0.69, 0.5, 0.78, 0.45, 0.81, 0.32, 0.8, 0.31, 0.84, 0.31, 0.88, 0.37, 0.9, 0.43, 0.9, 0.55,
        0.88, 0.59, 0.83, 0.64, 0.75, 0.65, 0.67, 0.66, 0.59, 0.66, 0.47, 0.66, 0.39, 0.66, 0.3, 0.66, 0.25,
        0.66, 0.2, 0.66, 0.15, 0.66, 0.1, 0.59, 0.1] + [0.5]*32,
    [0.22, 0.1, 0.22, 0.23, 0.22, 0.35, 0.22, 0.56, 0.22, 0.9, 0.26, 0.9, 0.32, 0.9, 0.32, 0.79, 0.32, 0.6,
        0.36, 0.56, 0.4, 0.51, 0.5, 0.66, 0.66, 0.9, 0.71, 0.9, 0.78, 0.9, 0.7, 0.78, 0.65, 0.71, 0.58, 0.61,
        0.47, 0.44, 0.57, 0.32, 0.6, 0.29, 0.67, 0.2, 0.76, 0.1, 0.69, 0.1, 0.63, 0.1, 0.52, 0.24, 0.32, 0.49,
        0.32, 0.28, 0.32, 0.1, 0.27, 0.1]+ [0.5]*32,
    [0.28, 0.1, 0.28, 0.14, 0.28, 0.21, 0.28, 0.27, 0.28, 0.36, 0.28, 0.44, 0.28, 0.51, 0.28, 0.59, 0.28,
        0.63, 0.28, 0.72, 0.28, 0.9, 0.41, 0.9, 0.48, 0.9, 0.59, 0.9, 0.72, 0.9, 0.72, 0.85, 0.72, 0.81, 0.63,
        0.81, 0.53, 0.81, 0.45, 0.81, 0.38, 0.81, 0.38, 0.63, 0.38, 0.55, 0.38, 0.44, 0.38, 0.36, 0.38, 0.29,
        0.38, 0.23, 0.38, 0.16, 0.38, 0.1, 0.32, 0.1] + [0.5]*32,
    [0.17, 0.15, 0.16, 0.29, 0.15, 0.44, 0.14, 0.59, 0.12, 0.9, 0.14, 0.9, 0.21, 0.9, 0.23, 0.61, 0.25,
        0.25, 0.35, 0.58, 0.45, 0.9, 0.48, 0.9, 0.53, 0.9, 0.64, 0.6, 0.76, 0.25, 0.77, 0.56, 0.79, 0.9, 0.82,
        0.9, 0.88, 0.9, 0.87, 0.65, 0.86, 0.53, 0.85, 0.38, 0.83, 0.15, 0.79, 0.15, 0.71, 0.15, 0.64, 0.35,
        0.49, 0.75, 0.4, 0.45, 0.29, 0.15, 0.23, 0.15] + [0.5]*32,
    [0.2, 0.1, 0.2, 0.33, 0.2, 0.49, 0.2, 0.67, 0.2, 0.9, 0.24, 0.9, 0.3, 0.9, 0.3, 0.7, 0.3, 0.53, 0.3,
        0.42, 0.29, 0.23, 0.4, 0.42, 0.46, 0.52, 0.52, 0.62, 0.7, 0.9, 0.74, 0.9, 0.8, 0.9, 0.8, 0.76, 0.8,
        0.53, 0.8, 0.34, 0.8, 0.1, 0.75, 0.1, 0.7, 0.1, 0.7, 0.2, 0.7, 0.35, 0.7, 0.47, 0.71, 0.76, 0.53,
        0.45, 0.31, 0.1, 0.26, 0.1] + [0.5]*32,
    [0.45, 0.1, 0.35, 0.13, 0.29, 0.16, 0.23, 0.22, 0.19, 0.28, 0.16, 0.34, 0.14, 0.42, 0.14, 0.5, 0.14,
        0.57, 0.16, 0.66, 0.18, 0.72, 0.23, 0.79, 0.28, 0.84, 0.34, 0.87, 0.43, 0.9, 0.54, 0.9, 0.62, 0.88,
        0.69, 0.85, 0.75, 0.8, 0.8, 0.73, 0.83, 0.65, 0.85, 0.53, 0.86, 0.49, 0.86, 0.41, 0.82, 0.31, 0.79,
        0.25, 0.75, 0.2, 0.7, 0.15, 0.62, 0.1, 0.54, 0.1, 0.38, 0.21, 0.26, 0.4, 0.29, 0.68, 0.41, 0.81,
        0.61, 0.79, 0.73, 0.61, 0.75, 0.4, 0.65, 0.23] + [0.5]*16,
    [0.26, 0.12, 0.26, 0.24, 0.26, 0.33, 0.26, 0.39, 0.26, 0.51, 0.26, 0.59, 0.26, 0.65, 0.26, 0.71, 0.26,
        0.9, 0.29, 0.9, 0.36, 0.9, 0.36, 0.81, 0.36, 0.74, 0.36, 0.68, 0.36, 0.58, 0.43, 0.58, 0.5, 0.59,
        0.55, 0.56, 0.6, 0.54, 0.67, 0.5, 0.7, 0.47, 0.74, 0.42, 0.74, 0.36, 0.74, 0.28, 0.7, 0.21, 0.68,
        0.17, 0.61, 0.13, 0.5, 0.1, 0.44, 0.1, 0.38, 0.1, 0.36, 0.19, 0.36, 0.29, 0.36, 0.49, 0.46, 0.51,
        0.57, 0.48, 0.64, 0.37, 0.6, 0.23, 0.53, 0.19] + [0.5]*16,
    [0.36, 0.13, 0.27, 0.21, 0.2, 0.35, 0.18, 0.44, 0.18, 0.5, 0.2, 0.58, 0.23, 0.65, 0.28, 0.71, 0.33,
        0.76, 0.38, 0.79, 0.5, 0.81, 0.59, 0.84, 0.69, 0.87, 0.78, 0.9, 0.8, 0.87, 0.82, 0.83, 0.73, 0.81,
        0.62, 0.78, 0.7, 0.73, 0.75, 0.68, 0.79, 0.6, 0.8, 0.5, 0.81, 0.43, 0.8, 0.36, 0.76, 0.25, 0.72,
        0.19, 0.67, 0.13, 0.6, 0.1, 0.52, 0.1, 0.45, 0.1, 0.4, 0.2, 0.31, 0.3, 0.28, 0.47, 0.34, 0.66, 0.49,
        0.73, 0.66, 0.66, 0.73, 0.43, 0.67, 0.26] + [0.5]*16,
    [0.25, 0.11, 0.25, 0.3, 0.25, 0.44, 0.25, 0.59, 0.24, 0.9, 0.27, 0.9, 0.35, 0.9, 0.35, 0.82, 0.35,
        0.71, 0.35, 0.66, 0.35, 0.55, 0.47, 0.55, 0.54, 0.58, 0.59, 0.65, 0.6, 0.73, 0.63, 0.8, 0.66, 0.9,
        0.7, 0.9, 0.76, 0.9, 0.72, 0.74, 0.7, 0.64, 0.67, 0.59, 0.59, 0.52, 0.69, 0.45, 0.74, 0.35, 0.72,
        0.22, 0.67, 0.15, 0.63, 0.13, 0.54, 0.1, 0.43, 0.1, 0.35, 0.18, 0.34, 0.29, 0.35, 0.47, 0.47, 0.47,
        0.61, 0.41, 0.64, 0.32, 0.61, 0.24, 0.53, 0.18] + [0.5]*16,
    [0.71, 0.14, 0.61, 0.1, 0.5, 0.1, 0.42, 0.12, 0.35, 0.16, 0.3, 0.24, 0.28, 0.31, 0.32, 0.43, 0.38,
        0.48, 0.5, 0.54, 0.59, 0.59, 0.65, 0.69, 0.59, 0.79, 0.46, 0.81, 0.29, 0.78, 0.27, 0.86, 0.37, 0.9,
        0.49, 0.9, 0.64, 0.87, 0.71, 0.79, 0.73, 0.71, 0.74, 0.65, 0.7, 0.56, 0.62, 0.49, 0.5, 0.45, 0.41,
        0.38, 0.43, 0.21, 0.51, 0.19, 0.68, 0.22, 0.71, 0.14] + [0.5]*32,
    [0.2, 0.1, 0.2, 0.15, 0.2, 0.19, 0.31, 0.19, 0.45, 0.19, 0.45, 0.3, 0.45, 0.35, 0.45, 0.42, 0.45, 0.5,
        0.45, 0.58, 0.45, 0.66, 0.45, 0.73, 0.45, 0.9, 0.5, 0.9, 0.55, 0.9, 0.55, 0.74, 0.55, 0.67, 0.55,
        0.59, 0.55, 0.53, 0.55, 0.43, 0.55, 0.37, 0.55, 0.29, 0.55, 0.19, 0.68, 0.19, 0.8, 0.19, 0.8, 0.14,
        0.8, 0.1, 0.65, 0.1, 0.51, 0.1, 0.34, 0.1] + [0.5]*32,
    [0.2, 0.1, 0.2, 0.32, 0.2, 0.41, 0.2, 0.55, 0.21, 0.67, 0.25, 0.78, 0.3, 0.86, 0.43, 0.9, 0.57, 0.89,
        0.69, 0.83, 0.74, 0.76, 0.79, 0.59, 0.8, 0.44, 0.8, 0.29, 0.8, 0.1, 0.72, 0.1, 0.68, 0.1, 0.68,
        0.23, 0.68, 0.36, 0.68, 0.54, 0.64, 0.64, 0.62, 0.74, 0.51, 0.78, 0.37, 0.77, 0.32, 0.68, 0.3, 0.55,
        0.3, 0.45, 0.3, 0.24, 0.3, 0.1, 0.26, 0.1] + [0.5]*32,
    [0.18, 0.1, 0.24, 0.31, 0.28, 0.41, 0.31, 0.52, 0.33, 0.58, 0.37, 0.7, 0.44, 0.9, 0.49, 0.9, 0.55, 0.9,
        0.59, 0.79, 0.64, 0.64, 0.68, 0.53, 0.73, 0.4, 0.77, 0.29, 0.84, 0.1, 0.77, 0.1, 0.73, 0.1, 0.69,
        0.2, 0.67, 0.26, 0.64, 0.35, 0.61, 0.43, 0.57, 0.57, 0.5, 0.79, 0.45, 0.63, 0.42, 0.5, 0.39, 0.42,
        0.35, 0.3, 0.33, 0.22, 0.29, 0.1, 0.24, 0.1] + [0.5]*32,
    [0.11, 0.18, 0.15, 0.36, 0.18, 0.47, 0.21, 0.58, 0.27, 0.82, 0.31, 0.82, 0.36, 0.82, 0.44, 0.54, 0.5,
        0.27, 0.56, 0.53, 0.63, 0.82, 0.67, 0.82, 0.72, 0.82, 0.78, 0.61, 0.81, 0.49, 0.84, 0.38, 0.9, 0.18,
        0.87, 0.18, 0.81, 0.18, 0.76, 0.38, 0.68, 0.73, 0.63, 0.5, 0.55, 0.18, 0.5, 0.18, 0.46, 0.18, 0.4,
        0.41, 0.32, 0.73, 0.27, 0.48, 0.2, 0.18, 0.16, 0.18] + [0.5]*32,
    [0.21, 0.1, 0.29, 0.23, 0.33, 0.3, 0.37, 0.36, 0.45, 0.5, 0.4, 0.58, 0.35, 0.66, 0.31, 0.72, 0.2, 0.9,
        0.27, 0.9, 0.32, 0.9, 0.4, 0.75, 0.51, 0.56, 0.59, 0.71, 0.7, 0.9, 0.76, 0.9, 0.82, 0.9, 0.76, 0.79,
        0.69, 0.68, 0.64, 0.6, 0.56, 0.49, 0.69, 0.3, 0.82, 0.1, 0.75, 0.1, 0.7, 0.1, 0.62, 0.23, 0.52,
        0.42, 0.42, 0.25, 0.33, 0.1, 0.28, 0.1] + [0.5]*32,
    [0.2, 0.1, 0.26, 0.22, 0.32, 0.32, 0.36, 0.4, 0.45, 0.56, 0.45, 0.63, 0.45, 0.71, 0.45, 0.78, 0.45,
        0.9, 0.49, 0.9, 0.56, 0.9, 0.56, 0.84, 0.56, 0.72, 0.56, 0.64, 0.56, 0.56, 0.62, 0.45, 0.67, 0.37,
        0.73, 0.27, 0.82, 0.1, 0.77, 0.1, 0.71, 0.1, 0.66, 0.18, 0.63, 0.26, 0.6, 0.31, 0.51, 0.48, 0.45,
        0.37, 0.4, 0.27, 0.37, 0.2, 0.32, 0.1, 0.27, 0.1] + [0.5]*32,
    [0.25, 0.1, 0.25, 0.14, 0.25, 0.19, 0.31, 0.19, 0.41, 0.19, 0.53, 0.19, 0.65, 0.19, 0.53, 0.37, 0.42,
        0.53, 0.33, 0.66, 0.21, 0.84, 0.21, 0.87, 0.21, 0.9, 0.33, 0.9, 0.48, 0.9, 0.62, 0.9, 0.79, 0.9,
        0.79, 0.87, 0.79, 0.81, 0.7, 0.81, 0.6, 0.81, 0.5, 0.81, 0.34, 0.81, 0.43, 0.68, 0.51, 0.56, 0.64,
        0.38, 0.79, 0.16, 0.79, 0.13, 0.79, 0.1, 0.58, 0.1] + [0.5]*32
]

#######################################################################################################################

chinese_topology = [8, 10, 4, 6, 4, 4, 9, 4]
chinese_letter_templates = [
    [0.08, 0.33, 0.1, 0.36, 0.12, 0.38, 0.17, 0.33, 0.21, 0.3, 0.21, 0.43, 0.21, 0.61, 0.25, 0.61, 0.28, 0.61, 0.28,
        0.41, 0.28, 0.21, 0.3, 0.18, 0.35, 0.13, 0.32, 0.09, 0.29, 0.08, 0.18, 0.21, 0.32, 0.32, 0.43, 0.21, 0.5, 0.08,
        0.53, 0.09, 0.57, 0.12, 0.53, 0.16, 0.5, 0.2, 0.7, 0.2, 0.88, 0.2, 0.85, 0.27, 0.82, 0.34, 0.8, 0.34, 0.75,
        0.32, 0.78, 0.29, 0.79, 0.26, 0.63, 0.25, 0.47, 0.25, 0.41, 0.32, 0.37, 0.36, 0.35, 0.33, 0.33, 0.53, 0.41,
        0.44, 0.47, 0.36, 0.5, 0.38, 0.53, 0.39, 0.46, 0.49, 0.38, 0.57, 0.37, 0.55, 0.5, 0.54, 0.52, 0.58, 0.53, 0.61,
        0.6, 0.59, 0.64, 0.55, 0.65, 0.41, 0.65, 0.3, 0.62, 0.3, 0.58, 0.3, 0.59, 0.43, 0.59, 0.53, 0.55, 0.53, 0.71,
        0.39, 0.78, 0.47, 0.84, 0.54, 0.87, 0.52, 0.9, 0.49, 0.83, 0.42, 0.76, 0.35, 0.73, 0.37, 0.71, 0.66, 0.73, 0.63,
        0.76, 0.61, 0.85, 0.71, 0.9, 0.77, 0.87, 0.8, 0.85, 0.82, 0.78, 0.73, 0.67, 0.77, 0.72, 0.78, 0.75, 0.79, 0.72,
        0.86, 0.67, 0.89, 0.52, 0.9, 0.38, 0.9, 0.34, 0.86, 0.31, 0.81, 0.31, 0.73, 0.31, 0.65, 0.36, 0.65, 0.38, 0.65,
        0.38, 0.74, 0.39, 0.82, 0.5, 0.84, 0.62, 0.84, 0.66, 0.81, 0.2, 0.65, 0.23, 0.67, 0.26, 0.68, 0.2, 0.79, 0.16,
        0.87, 0.12, 0.85, 0.09, 0.83, 0.16, 0.73]
]
if __name__ == '__main__':
    # Combine the given control points and the additional 16 points (0.5)



    control_points_tensor = letter_templates[23]
    control_points_tensor = torch.tensor(control_points_tensor).view(-1, 2)

    n_outer_curves = 15
    n_inner_curves = 4

    outer_shape_control_points = control_points_tensor[:n_outer_curves * 3]
    inner_shape_control_points = control_points_tensor[n_outer_curves * 3:]

    outer_reshaped = outer_shape_control_points.view(n_outer_curves, 3, 2)
    inner_reshaped = inner_shape_control_points.view(n_inner_curves, 3, 2)

    reshaped_control_points = torch.cat((outer_reshaped, inner_reshaped), dim=0)

    # Call the modified function to plot the letter 'A'
    plotQuadraticSpline(reshaped_control_points, title="Letter A")


