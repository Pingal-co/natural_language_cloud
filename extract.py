import re
import probablepeople as pp
import usaddress
from date_parse import datetime_parsing
import json
# pip install parserator probablepeople usaddress 

# based on https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
date             = re.compile(u'(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE)
time             = re.compile(u'\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
phone            = re.compile(u'''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''')
phones_with_exts = re.compile(u'((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE)
link             = re.compile(u'(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)', re.IGNORECASE)
email            = re.compile(u"([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE)
ip               = re.compile(u'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE)
ipv6             = re.compile(u'\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*', re.VERBOSE|re.IGNORECASE|re.DOTALL)
price            = re.compile(u'[$]\s?[+-]?[0-9]{1,3}(?:(?:,?[0-9]{3}))*(?:\.[0-9]{1,2})?')
hex_color        = re.compile(u'(#(?:[0-9a-fA-F]{8})|#(?:[0-9a-fA-F]{3}){1,2})\\b')
credit_card      = re.compile(u'((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])')
btc_address      = re.compile(u'(?<![a-km-zA-HJ-NP-Z0-9])[13][a-km-zA-HJ-NP-Z0-9]{26,33}(?![a-km-zA-HJ-NP-Z0-9])')
street_address   = re.compile(u'\d{1,4} [\w\s]{1,20}(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)', re.IGNORECASE)
zip_code = re.compile(r'\b\d{5}(?:[-\s]\d{4})?\b')
po_box = re.compile(r'P\.? ?O\.? Box \d+', re.IGNORECASE)

# 
def extract_info(text):
    output = {
        'date': [x.strip() for x in date.findall(text)],
        'time': [x.strip() for x in time.findall(text)],
        'datetime': datetime_parsing(text),
        'phone': [x.strip() for x in phone.findall(text)],
        'phones_with_exts': [x.strip() for x in phones_with_exts.findall(text)],
        'link': [x.strip() for x in link.findall(text) if x in text.split(" ")],
        'email': [x.strip() for x in email.findall(text)],
        'ip': [x.strip() for x in ip.findall(text)],
        'price' : [x.strip() for x in price.findall(text)],
        'credit_card': [x.strip() for x in credit_card.findall(text)],
        'street_address': [x.strip() for x in street_address.findall(text)],
        'zip_code': [x.strip() for x in zip_code.findall(text)]
    }
    return output

def add_address(text):    
    # check if text only contains an address field
    if is_address('text'):
        return extract_address(text)["data"]
    
# parsing using CRF probabilistic parser
def extract_entity(text):
    (tags, entity_type) = pp.tag(text)
    # fix the schema output
    output = {'type': '', 'data': {}}
    # person = {'type' : 'Person', 'data': {prefix': '', 'first_name': '', 'last_name': '', 'nick_name': ''}}
    # organization = {'type' : 'Organization', 'data': {name': '', 'legal_type' : ''}}

    valid_corporate_keys = ['CorporationName', 'CorporationLegalType']
    valid_person_keys = ['PrefixMarital', 'GivenName', 'Surname', 'Nickname']

 
    for key in tags:
        if entity_type == 'Corporation' and key in valid_corporate_keys:
            output['type'] = 'Organization'
            if key == 'CorporationName': output['data']['name'] = tags[key] 
            if key == 'CorporationLegalType': output['data']['legal_type'] = tags[key]  
        if entity_type == 'Person' and key in valid_person_keys:
            output['type'] = 'Person'
            if key == 'PrefixMarital': output['data']['prefix'] = tags[key]  
            if key == 'GivenName': output['data']['first_name'] = tags[key]  
            if key == 'Surname': output['data']['last_name'] = tags[key]  
            if key == 'Nickname': output['data']['nick_name'] = tags[key]  
       
    return output
        
def extract_address(text):
    (tags, entity_type) = usaddress.tag(text)
    output = {'type': '', 'data': {}}
    # address = {'type' : 'Address', 'data': {'street': '', 'occupancy_unit' : '', city: '', state: '', country: 'USA'}}
    if 'Address' in entity_type:
        output['type'] = 'Address'
        output['data']['street'] = tags.get('AddressNumber', '') + ' ' + tags.get('StreetName', '') + ' ' + tags.get('StreetNamePostType', '') + ' ' + tags.get('StreetNamePostDirectional', '')
        output['data']['building_unit'] = tags.get('BuildingName', '') + ' ' + tags.get('OccupancyType', '') + ' ' + tags.get('OccupancyIdentifier', '')
        output['data']['city'] = tags.get('PlaceName', '')
        output['data']['state'] = tags.get('StateName', '')
        output['data']['country'] = 'USA'
        output['data']['zip'] = tags.get('ZipCode', '')
    
    return output   


def is_address(text):
    info = extract_info(text)
    invalid_keys = ['date', 'time', 'phone', 'phones_with_exts', 'link', 'email', 'ip', 'price', 'credit_card' ]
    
    for k,v in info.items():
        if k in invalid_keys and v:
            return False

    if info["street_address"]:
        return True
    
    return False

def has_email(text):
    email = [x.strip() for x in email.findall(text)]
    if email:
        return True

    return False

def has_phone(text):
    phone = [x.strip() for x in phone.findall(text)]
    if phone:
        return True

    return False


