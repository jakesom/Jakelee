classname = {0: '01-Nimitz Aircraft Carrier', 1: '06-Barracks Ship', 2: '05-Container Ship',
                            3: '06-Fishing Vessel', 4: '06-Henry J. Kaiser-class replenishment oiler',
                            5: '06-Other Warship', 6: '06-Yacht', 7: '06-Freedom-class littoral combat ship',
                            8: '02-Arleigh Burke-class Destroyer', 9: '06-Lewis and Clark-class dry cargo ship',
                            10: '06-Towing vessel', 11: '06-unknown', 12: '06-Powhatan-class tugboat',
                            13: '06-Barge', 14: '02-055-destroyer', 15: '02-052D-destroyer', 16: '06-USNS Bob Hope',
                            17: '06-USNS Montford Point', 18: '06-Bunker', 19: '06-Ticonderoga-class cruiser',
                            20: '03-Oliver Hazard Perry-class frigate',
                            21: '06-Sacramento-class fast combat support ship', 22: '06-Submarine',
                            23: '06-Emory S. Land-class submarine tender', 24: '06-Hatakaze-class destroyer',
                            25: '02-Murasame-class destroyer', 26: '06-Whidbey Island-class dock landing ship',
                            27: '06-Hiuchi-class auxiliary multi-purpose support ship', 28: '06-USNS Spearhead',
                            29: '02-Hyuga-class helicopter destroyer', 30: '02-Akizuki-class destroyer',
                            31: '05-Bulk carrier', 32: '02-Kongo-class destroyer', 33: '06-Northampton-class tug',
                            34: '05-Sand Carrier', 35: '06-Iowa-class battle ship',
                            36: '06-Independence-class littoral combat ship',
                            37: '06-Tarawa-class amphibious assault ship', 38: '04-Cyclone-class patrol ship',
                            39: '06-Wasp-class amphibious assault ship', 40: '06-074-landing ship',
                            41: '06-056-corvette', 42: '06-721-transport boat', 43: '06-037II-missile boat',
                            44: '06-Traffic boat', 45: '06-037-submarine chaser', 46: '06-unknown auxiliary ship',
                            47: '06-072III-landing ship', 48: '06-636-hydrographic survey ship',
                            49: '06-272-icebreaker', 50: '06-529-Minesweeper', 51: '03-053H2G-frigate',
                            52: '06-909A-experimental ship', 53: '06-909-experimental ship',
                            54: '06-037-hospital ship', 55: '06-Tuzhong Class Salvage Tug',
                            56: '02-022-missile boat', 57: '02-051-destroyer', 58: '03-054A-frigate',
                            59: '06-082II-Minesweeper', 60: '03-053H1G-frigate', 61: '06-Tank ship',
                            62: '02-Hatsuyuki-class destroyer', 63: '06-Sugashima-class minesweepers',
                            64: '06-YG-203 class yard gasoline oiler',
                            65: '06-Hayabusa-class guided-missile patrol boats', 66: '06-JS Chihaya',
                            67: '06-Kurobe-class training support ship', 68: '03-Abukuma-class destroyer escort',
                            69: '06-Uwajima-class minesweepers', 70: '06-Osumi-class landing ship',
                            71: '06-Hibiki-class ocean surveillance ships',
                            72: '06-JMSDF LCU-2001 class utility landing crafts', 73: '02-Asagiri-class Destroyer',
                            74: '06-Uraga-class Minesweeper Tender', 75: '06-Tenryu-class training support ship',
                            76: '06-YW-17 Class Yard Water', 77: '02-Izumo-class helicopter destroyer',
                            78: '06-Towada-class replenishment oilers', 79: '02-Takanami-class destroyer',
                            80: '06-YO-25 class yard oiler', 81: '06-891A-training ship', 82: '03-053H3-frigate',
                            83: '06-922A-Salvage lifeboat', 84: '06-680-training ship', 85: '06-679-training ship',
                            86: '06-072A-landing ship', 87: '06-072II-landing ship',
                            88: '06-Mashu-class replenishment oilers', 89: '06-903A-replenishment ship',
                            90: '06-815A-spy ship', 91: '06-901-fast combat support ship',
                            92: '06-Xu Xiake barracks ship', 93: '06-San Antonio-class amphibious transport dock',
                            94: '06-908-replenishment ship', 95: '02-052B-destroyer',
                            96: '06-904-general stores issue ship', 97: '02-051B-destroyer',
                            98: '06-925-Ocean salvage lifeboat', 99: '06-904B-general stores issue ship',
                            100: '06-625C-Oceanographic Survey Ship', 101: '06-071-amphibious transport dock',
                            102: '02-052C-destroyer', 103: '06-635-hydrographic Survey Ship',
                            104: '06-926-submarine support ship', 105: '06-917-lifeboat',
                            106: '06-Mercy-class hospital ship',
                            107: '06-Lewis B. Puller-class expeditionary mobile base ship',
                            108: '06-Avenger-class mine countermeasures ship', 109: '02-Zumwalt-class destroyer',
                            110: '06-920-hospital ship', 111: '02-052-destroyer', 112: '06-054-frigate',
                            113: '02-051C-destroyer', 114: '06-903-replenishment ship', 115: '06-073-landing ship',
                            116: '06-074A-landing ship', 117: '06-North Transfer 990',
                            118: '01-001-aircraft carrier', 119: '06-905-replenishment ship',
                            120: '06-Hatsushima-class minesweeper', 121: '01-Forrestal-class Aircraft Carrier',
                            122: '01-Kitty Hawk class aircraft carrier', 123: '06-Blue Ridge class command ship',
                            124: '06-081-Minesweeper', 125: '06-648-submarine repair ship',
                            126: '06-639A-Hydroacoustic measuring ship', 127: '06-JS Kurihama', 128: '06-JS Suma',
                            129: '06-Futami-class hydro-graphic survey ships', 130: '06-Yaeyama-class minesweeper',
                            131: '06-815-spy ship', 132: '02-Sovremenny-class destroyer'}

new_classname = {}
for key, value in classname.items():
    if value.startswith('01'):
        new_classname[key] = 'aircraft_carrier'
    elif value.startswith('02'):
        new_classname[key] = 'destroyer'
    elif value.startswith('03'):
        new_classname[key] = 'frigate'
    elif value.startswith('04'):
        new_classname[key] = 'patrol_ship'
    elif value.startswith('05'):
        new_classname[key] = 'cargo_ship'
    elif value.startswith('06'):
        new_classname[key] = 'other_ship'

print(new_classname)
