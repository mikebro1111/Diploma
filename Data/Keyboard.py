from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

btnMain = KeyboardButton('Menu')

#main menu
class Keyboard:
    def create_type(clr) -> ReplyKeyboardMarkup:
        btnType = KeyboardButton('Type of investor')
        btnGoals = KeyboardButton('Financial goals')
        btnRisk = KeyboardButton('Percentage of risk')
        btnTerm = KeyboardButton('Investment term')
        btnRegion = KeyboardButton('Region selection')
        mainMenu = ReplyKeyboardMarkup(resize_keyboard= True).add(btnType, btnGoals, btnRisk, btnTerm, btnRegion)
        return Keyboard

    #In Type
    def create_type(clr) -> ReplyKeyboardMarkup:
        btnBeginner = KeyboardButton ('Beginner')
        btnIntermediate = KeyboardButton ('Intermediate')
        btnExperienced = KeyboardButton ('Experienced')
        typeMenu = ReplyKeyboardMarkup (btnBeginner, btnIntermediate, btnExperienced, btnMain)
        return Keyboard

    #In Goals
    def create_type(clr) -> ReplyKeyboardMarkup:
        btnSaving = KeyboardButton ('Saving money')
        btnCapital = KeyboardButton ('Capital increase')
        btnIncome = KeyboardButton ('Income for retirement')
        btnEducation = KeyboardButton ('Education of children')
        btnPurchase = KeyboardButton ('c of housing')
        goalsMenu = ReplyKeyboardMarkup (btnSaving, btnCapital, btnIncome, btnEducation, btnPurchase, btnMain)
        return Keyboard

    #In Risk
    def create_type(clr) -> ReplyKeyboardMarkup:
        btn5 = KeyboardButton ('<5%')
        btn10 = KeyboardButton ('5-10%')
        btn20 = KeyboardButton ('10-20%')
        btn30 = KeyboardButton ('20-30%')
        btn40 = KeyboardButton ('30-40%')
        btn50 = KeyboardButton ('40-50%')
        btn60 = KeyboardButton ('50-60%')
        btn70 = KeyboardButton ('60-70%')
        btn80 = KeyboardButton ('70-80%')
        btn90 = KeyboardButton ('80-90%')
        btn100 = KeyboardButton ('90-100%')
        btnConservative = KeyboardButton ('Conservative')
        btnModerate = KeyboardButton ('Moderate')
        btnAggressive = KeyboardButton ('Aggressive')
        riskMenu = ReplyKeyboardMarkup (btn5, btn10, btn20, btn30, btn40, btn50, btn60, btn70, btn80, btn90, btn100, btnConservative, btnModerate, btnAggressive, btnMain)
        return Keyboard
    
    #In Term
    def create_type(clr) -> ReplyKeyboardMarkup:
        btnShort = KeyboardButton ('Short term (up to six months)')
        btnShort1 = KeyboardButton ('Short term (1 year)')
        btnShort2 = KeyboardButton ('Short term (1 and a half years)')
        btnShort3 = KeyboardButton ('Short term (2 years)')
        btnShort4 = KeyboardButton ('Short term (2 and a half years)')
        btnIntermediate = KeyboardButton ('Average term (up to 3 years)')
        btnIntermediate1 = KeyboardButton ('Average term (3 and a half years)')
        btnIntermediate2 = KeyboardButton ('Average term (4 years)')
        btnIntermediate3 = KeyboardButton ('Average term (4 and a half years)')
        btnExperienced = KeyboardButton ('Long term (5+ years)')
        termMenu = ReplyKeyboardMarkup (btnShort, btnShort1, btnShort2, btnShort3, btnShort4, btnIntermediate, btnIntermediate1, btnIntermediate2, btnIntermediate3, btnExperienced, btnMain)
        return Keyboard

    #In Region
    def create_type(clr) -> ReplyKeyboardMarkup:
        btnAsia = KeyboardButton ('Asia')
        btnEurope = KeyboardButton ('Europe')
        btnAmerica = KeyboardButton ('America')
        btnAfrica = KeyboardButton ('Africa')
        regionMenu = ReplyKeyboardMarkup (btnAsia, btnEurope, btnAmerica, btnAfrica, btnMain)
        return Keyboard

