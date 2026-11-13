from datetime import date

from .database import SessionLocal
from .models import Disease, ManagementGuideline


# ============================================================
# VERIFIED AGRICULTURAL KNOWLEDGE BASE
# ============================================================
#
# AI detects the condition.
# Verified agricultural sources provide information.
# Farmer/agricultural expert makes the final decision.
#
# This database contains only conditions for which
# agricultural information has been verified.
# ============================================================


diseases = [

    # ========================================================
    # 1. CORN - CERCOSPORA LEAF SPOT
    # ========================================================

    {
        "model_class": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "crop": "Corn",
        "disease_name": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "scientific_name": "Cercospora zeae-maydis",

        "description": (
            "A fungal foliar disease of corn that can become more "
            "severe under warm, humid conditions and where infected "
            "corn residue remains in the field."
        ),

        "symptoms": (
            "Gray to tan, rectangular lesions that are generally "
            "limited by the leaf veins. Lesions commonly develop "
            "first on lower leaves and can expand under favorable "
            "conditions."
        ),

        "prevention": (
            "Use hybrids with appropriate resistance, rotate corn "
            "with non-host crops, manage infected crop residue, "
            "and scout fields regularly for disease development."
        ),

        "source": (
            "NC State Extension — Gray Leaf Spot in Corn"
        ),

        "management": (
            "Use resistant or partially resistant hybrids where "
            "appropriate. Crop rotation and management of infected "
            "residue can reduce disease pressure. Field scouting "
            "should guide management decisions."
        )
    },


    # ========================================================
    # 2. CORN - COMMON RUST
    # ========================================================

    {
        "model_class": "Corn___Common_rust",
        "crop": "Corn",
        "disease_name": "Common Rust",
        "scientific_name": "Puccinia sorghi",

        "description": (
            "A fungal rust disease of corn that is favored by "
            "cool, wet conditions."
        ),

        "symptoms": (
            "Rust-colored to dark brown elongated pustules can "
            "occur on both surfaces of corn leaves. Pustules "
            "darken as they age."
        ),

        "prevention": (
            "Use resistant corn hybrids where appropriate and "
            "monitor fields when environmental conditions favor "
            "disease development."
        ),

        "source": (
            "University of Minnesota Extension — Common Rust on Corn"
        ),

        "management": (
            "Resistant hybrids are an important management option. "
            "Fungicide decisions should be based on disease "
            "development, crop stage, local recommendations and "
            "product labels."
        )
    },


    # ========================================================
    # 3. GRAPE - BLACK ROT
    # ========================================================

    {
        "model_class": "Grape___Black_rot",
        "crop": "Grape",
        "disease_name": "Black Rot",
        "scientific_name": "Guignardia bidwellii",

        "description": (
            "A fungal disease of grapevines that can infect green "
            "tissues and is particularly damaging to fruit."
        ),

        "symptoms": (
            "Leaves may develop reddish-brown circular lesions "
            "with small black fruiting structures. Infected "
            "berries become dark and may shrivel into hard black "
            "mummies."
        ),

        "prevention": (
            "Remove and properly manage mummified berries and "
            "infected plant material. Maintain vineyard sanitation "
            "and consider appropriate resistant varieties."
        ),

        "source": (
            "Penn State Extension — Grape Disease: Black Rot"
        ),

        "management": (
            "Remove infected prunings and mummified berries before "
            "new growth. Maintain vineyard sanitation and use "
            "locally recommended disease-management practices "
            "when disease risk is high."
        )
    },


    # ========================================================
    # 4. TOMATO - BACTERIAL SPOT
    # ========================================================

    {
        "model_class": "Tomato___Bacterial_spot",
        "crop": "Tomato",
        "disease_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas species complex",

        "description": (
            "A bacterial disease affecting tomato foliage, stems "
            "and fruit, especially under warm, humid and wet "
            "conditions."
        ),

        "symptoms": (
            "Small brown circular leaf spots may develop with "
            "yellow halos. Spots can also occur on stems and fruit. "
            "Fruit lesions may be slightly raised and scabby."
        ),

        "prevention": (
            "Use healthy seed and transplants, avoid overhead "
            "irrigation, keep foliage dry, provide adequate plant "
            "spacing, sanitize tools, and remove or bury crop "
            "debris after the season."
        ),

        "source": (
            "University of Minnesota Extension — "
            "Bacterial Spot of Tomato and Pepper"
        ),

        "management": (
            "Focus on clean planting material, moisture management, "
            "sanitation and field hygiene. Avoid working with plants "
            "when foliage is wet because the pathogen can spread "
            "through water, tools and handling."
        )
    },


    # ========================================================
    # 5. TOMATO - EARLY BLIGHT
    # ========================================================

    {
        "model_class": "Tomato___Early_blight",
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "scientific_name": "Alternaria solani",

        "description": (
            "A fungal disease that commonly begins on older lower "
            "tomato leaves and can also affect stems and fruit."
        ),

        "symptoms": (
            "Dark brown leaf spots, often with concentric "
            "target-like rings, frequently beginning on older "
            "lower leaves. Stem and fruit lesions may also occur."
        ),

        "prevention": (
            "Improve airflow, keep foliage dry, use mulch to reduce "
            "soil splashing, remove infected leaves when appropriate, "
            "manage weeds and practice crop rotation."
        ),

        "source": (
            "University of Connecticut Extension — Early Blight of Tomato"
        ),

        "management": (
            "Use cultural practices that reduce leaf wetness and "
            "disease spread. Remove infected plant material "
            "appropriately and follow locally approved "
            "disease-management recommendations."
        )
    },


    # ========================================================
    # 6. TOMATO - LATE BLIGHT
    # ========================================================

    {
        "model_class": "Tomato___Late_blight",
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "scientific_name": "Phytophthora infestans",

        "description": (
            "A potentially destructive disease of tomato and "
            "potato caused by a water mold."
        ),

        "symptoms": (
            "Disease can cause rapidly developing lesions on "
            "foliage and other plant tissues. Disease development "
            "is strongly influenced by environmental conditions."
        ),

        "prevention": (
            "Scout plants regularly, maintain good airflow, keep "
            "foliage dry, avoid planting tomato, potato, pepper "
            "or eggplant repeatedly in the same location, and "
            "manage infected plant material."
        ),

        "source": (
            "University of Minnesota Extension — "
            "Late Blight of Tomato and Potato"
        ),

        "management": (
            "Early detection is important. Remove or manage "
            "diseased plant material and use locally recommended "
            "disease-management practices. Any pesticide use "
            "must follow the current product label."
        )
    },


    # ========================================================
    # 7. TOMATO - LEAF MOLD
    # ========================================================

    {
        "model_class": "Tomato___Leaf_Mold",
        "crop": "Tomato",
        "disease_name": "Leaf Mold",
        "scientific_name": "Passalora fulva",

        "description": (
            "A fungal disease that is particularly associated "
            "with tomato production under high-humidity greenhouse "
            "and high-tunnel conditions."
        ),

        "symptoms": (
            "Yellow spots may appear on the upper leaf surface "
            "while olive-green to brown velvety mold develops "
            "on the lower leaf surface. Severe infection can "
            "cause leaves to wither and die."
        ),

        "prevention": (
            "Reduce humidity, improve ventilation and air movement, "
            "use drip irrigation, avoid wetting foliage, sanitize "
            "structures and tools, and remove crop residue after "
            "harvest."
        ),

        "source": (
            "University of Minnesota Extension — Tomato Leaf Mold"
        ),

        "management": (
            "Prioritize humidity management, ventilation, sanitation "
            "and resistant varieties where suitable. Disease-management "
            "products should only be used according to current local "
            "recommendations and product labels."
        )
    },


    # ========================================================
    # 8. APPLE - APPLE SCAB
    # ========================================================

    {
        "model_class": "Apple___Apple_scab",
        "crop": "Apple",
        "disease_name": "Apple Scab",
        "scientific_name": "Venturia inaequalis",

        "description": (
            "A fungal disease favored by cool, wet spring conditions "
            "that can affect apple leaves and fruit."
        ),

        "symptoms": (
            "Olive-green spots appear on young leaves and can darken "
            "to greenish-black lesions. Leaves may turn brown and "
            "drop. Fruit can develop brown scab-like lesions, become "
            "distorted, and drop prematurely."
        ),

        "prevention": (
            "Remove fallen leaves and infected debris, prune to "
            "improve air circulation and sunlight penetration, "
            "avoid overhead irrigation, and consider resistant "
            "varieties."
        ),

        "source": (
            "University of Maryland Extension / "
            "University of New Hampshire Extension"
        ),

        "management": (
            "Disease management is most effective when preventive "
            "measures begin early in the season. Fungicide timing "
            "should follow local extension recommendations and "
            "current product labels."
        )
    },


    # ========================================================
    # 9. APPLE - BLACK ROT
    # ========================================================

    {
        "model_class": "Apple___Black_rot",
        "crop": "Apple",
        "disease_name": "Black Rot",
        "scientific_name": "Botryosphaeria obtusa",

        "description": (
            "A fungal disease of apple and other pome fruits that "
            "can infect fruit, branches, and other plant tissues."
        ),

        "symptoms": (
            "Fruit can develop rot following wounds, while dead "
            "twigs, branches and mummified fruit can serve as "
            "sources of infection. Infections may also develop "
            "after harvest and during storage."
        ),

        "prevention": (
            "Prune out cankers, dead branches and twigs; remove "
            "mummified fruit; properly dispose of infected plant "
            "material; maintain healthy trees and reduce plant stress."
        ),

        "source": (
            "Penn State Extension — Pome Fruit Disease: "
            "Black Rot and Frogeye Leaf Spot"
        ),

        "management": (
            "Sanitation is a primary management strategy. Where "
            "chemical control is used, timing should follow local "
            "orchard disease-management programs and current "
            "product labels."
        )
    },


    # ========================================================
    # 10. APPLE - CEDAR-APPLE RUST
    # ========================================================

    {
        "model_class": "Apple___Cedar_apple_rust",
        "crop": "Apple",
        "disease_name": "Cedar-Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",

        "description": (
            "A fungal rust requiring both apple and cedar or juniper "
            "hosts to complete its life cycle."
        ),

        "symptoms": (
            "Yellow to orange spots develop on apple leaves. As "
            "the disease progresses, aecia can form on the underside "
            "of leaves. Fruit may also develop lesions and become "
            "malformed."
        ),

        "prevention": (
            "Avoid planting susceptible apples near cedar or juniper "
            "where practical. Remove rust galls from nearby cedar "
            "or juniper during dormancy where feasible, and consider "
            "resistant apple varieties."
        ),

        "source": (
            "Oklahoma State University Extension / NC State Extension"
        ),

        "management": (
            "Management focuses on preventing infection through host "
            "separation, sanitation and appropriately timed "
            "disease-control programs. Follow local extension "
            "recommendations and current product labels for fungicide use."
        )
    },


    # ========================================================
    # 11. CHERRY - POWDERY MILDEW
    # ========================================================

    {
        "model_class": "Cherry___Powdery_mildew",
        "crop": "Cherry",
        "disease_name": "Powdery Mildew",
        "scientific_name": "Podosphaera clandestina",

        "description": (
            "A fungal disease affecting cherry trees, particularly "
            "in regions where environmental conditions favor "
            "powdery mildew."
        ),

        "symptoms": (
            "Powdery mildew produces characteristic white fungal "
            "growth on susceptible plant tissue and can affect "
            "young shoots and foliage."
        ),

        "prevention": (
            "Monitor trees regularly, maintain appropriate canopy "
            "management, and avoid excessive conditions that favor "
            "disease development. Irrigation and nitrogen management "
            "are important considerations in cherry powdery-mildew "
            "IPM programs."
        ),

        "source": (
            "Oregon State University Extension"
        ),

        "management": (
            "Local cherry disease-management programs may use "
            "cultural practices and fungicides where appropriate. "
            "Product selection and application should follow local "
            "recommendations and current product labels."
        )
    },


    # ========================================================
    # 12. CORN - NORTHERN LEAF BLIGHT
    # ========================================================

    {
        "model_class": "Corn___Northern_Leaf_Blight",
        "crop": "Corn",
        "disease_name": "Northern Leaf Blight",
        "scientific_name": "Exserohilum turcicum",

        "description": (
            "A fungal disease of corn favored by moderate "
            "temperatures and prolonged leaf moisture."
        ),

        "symptoms": (
            "Characteristic elongated, canoe-shaped tan lesions "
            "develop on leaves, usually beginning on lower leaves "
            "and progressing upward. Severe infections can cause "
            "extensive leaf blighting."
        ),

        "prevention": (
            "Use resistant hybrids and consider crop rotation and "
            "appropriate tillage practices. Timely planting may "
            "also help avoid conditions favorable to disease development."
        ),

        "source": (
            "Cornell University CALS — Northern Corn Leaf Blight"
        ),

        "management": (
            "Monitor fields for early disease development, particularly "
            "before silking. Fungicide decisions should consider "
            "disease development, hybrid susceptibility, crop stage "
            "and current local recommendations."
        )
    },

    # ========================================================
    # 13. GRAPE - ESCA (BLACK MEASLES)
    # ========================================================

    {
        "model_class": "Grape___Esca_(Black_Measles)",
        "crop": "Grape",
        "disease_name": "Esca (Black Measles)",
        "scientific_name": "Phaeoacremonium species and Phaeomoniella chlamydospora",

        "description": (
            "A grapevine trunk disease complex caused by multiple "
            "wood-infecting fungi."
        ),

        "symptoms": (
            "Leaves may develop interveinal striping that becomes "
            "necrotic. Symptoms can cause leaves to dry and drop "
            "prematurely. Berries may develop small dark spots "
            "surrounded by brown-purple rings."
        ),

        "prevention": (
            "Reduce opportunities for infection through careful "
            "vineyard sanitation and pruning practices. Protect "
            "pruning wounds where recommended and avoid spreading "
            "infected plant material."
        ),

        "source": (
            "UC Statewide Integrated Pest Management Program — "
            "Esca (Black Measles)"
        ),

        "management": (
            "Esca establishes permanent wood infections after "
            "infected pruning wounds and cannot be eradicated by "
            "fungicide applications once established. Follow local "
            "vineyard trunk-disease management recommendations."
        )
    },


    # ========================================================
    # 14. GRAPE - LEAF BLIGHT / ISARIOPSIS LEAF SPOT
    # ========================================================

    {
        "model_class": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "crop": "Grape",
        "disease_name": "Leaf Blight (Isariopsis Leaf Spot)",
        "scientific_name": "Pseudocercospora vitis",

        "description": (
            "A fungal leaf disease of grape caused by "
            "Pseudocercospora vitis, historically associated with "
            "the name Isariopsis leaf spot."
        ),

        "symptoms": (
            "Large irregular leaf spots may initially appear dull "
            "red to brown and become black and brittle as they age. "
            "The disease is generally more common late in the season."
        ),

        "prevention": (
            "Maintain good vineyard vigor and canopy management. "
            "Improve air circulation through appropriate pruning, "
            "thinning and spacing so plant tissues dry more quickly."
        ),

        "source": (
            "NC State Extension — North Carolina Winegrape Growers Guide"
        ),

        "management": (
            "Promote good air circulation and maintain vineyard "
            "sanitation. Standard grape disease-management programs "
            "may provide control; follow local recommendations and "
            "current product labels where fungicides are used."
        )
    },


    # ========================================================
    # 15. ORANGE - HUANGLONGBING / CITRUS GREENING
    # ========================================================

    {
        "model_class": "Orange___Haunglongbing_(Citrus_greening)",
        "crop": "Orange",
        "disease_name": "Huanglongbing (Citrus Greening)",
        "scientific_name": "Candidatus Liberibacter asiaticus",

        "description": (
            "A bacterial disease of citrus, commonly called "
            "Huanglongbing or citrus greening, that can severely "
            "reduce tree productivity and eventually kill infected trees."
        ),

        "symptoms": (
            "Leaves may show asymmetrical blotchy mottling. Infected "
            "trees may produce smaller, poorly colored and irregularly "
            "shaped fruit with a bitter taste. Twig dieback and "
            "premature fruit drop may also occur."
        ),

        "prevention": (
            "Use healthy, certified citrus planting material where "
            "available, follow applicable quarantine requirements, "
            "and prevent movement of regulated citrus plants and "
            "plant material from affected areas."
        ),

        "source": (
            "USDA APHIS — Citrus Greening and Asian Citrus Psyllid"
        ),

        "management": (
            "There is currently no cure for citrus greening. Management "
            "focuses on preventing introduction and spread, monitoring "
            "for the disease and its insect vector, and following "
            "applicable plant-health and quarantine guidance."
        )
    },


    # ========================================================
    # 16. PEACH - BACTERIAL SPOT
    # ========================================================

    {
        "model_class": "Peach___Bacterial_spot",
        "crop": "Peach",
        "disease_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas arboricola pv. pruni",

        "description": (
            "A bacterial disease of peach and other stone fruits "
            "that can affect leaves, fruit and young twigs."
        ),

        "symptoms": (
            "Leaf lesions begin as small angular spots and may become "
            "brown to black, sometimes developing a shot-hole appearance. "
            "Fruit can develop dark, sunken spots and may crack. Twig "
            "cankers can also occur."
        ),

        "prevention": (
            "Use resistant varieties where available and maintain "
            "vigorous orchard growth. Avoid practices that increase "
            "leaf wetness and follow local orchard disease-management "
            "recommendations."
        ),

        "source": (
            "Oklahoma State University Extension — "
            "Bacterial Spot of Stone Fruits"
        ),

        "management": (
            "Management should emphasize resistant varieties and "
            "appropriate orchard practices. Disease-control programs "
            "should be based on local conditions, extension guidance "
            "and current product labels."
        )
    },


    # ========================================================
    # 17. BELL PEPPER - BACTERIAL SPOT
    # ========================================================

    {
        "model_class": "Pepper,_bell___Bacterial_spot",
        "crop": "Bell Pepper",
        "disease_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas species complex",

        "description": (
            "A bacterial disease affecting pepper foliage, stems "
            "and fruit, particularly under warm, humid and wet conditions."
        ),

        "symptoms": (
            "Pepper leaves develop small brown circular spots. "
            "Spots may enlarge and cause affected foliage to look "
            "ragged. Brown spots can also develop on stems and fruit."
        ),

        "prevention": (
            "Use high-quality disease-free seed and transplants, "
            "avoid overhead irrigation, keep foliage dry, provide "
            "adequate plant spacing, sanitize tools and remove or "
            "bury crop debris after the season."
        ),

        "source": (
            "University of Minnesota Extension — "
            "Bacterial Spot of Tomato and Pepper"
        ),

        "management": (
            "Monitor plants during warm and humid periods. Use "
            "resistant varieties where available, avoid working "
            "with wet plants, maintain field sanitation and rotate "
            "away from tomato and pepper where appropriate. Any "
            "pesticide use must follow current product labels."
        )
    },

    {
        "model_class": "Potato___Early_blight",
        "crop": "Potato",
        "disease_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "description": "A fungal disease that commonly affects potato foliage and tubers.",
        "symptoms": "Brown to dark lesions with concentric rings, often beginning on older leaves; severe infection can cause premature leaf death.",
        "prevention": "Use healthy seed, maintain good field sanitation, rotate crops, and reduce conditions that favor prolonged leaf wetness.",
        "source": "University of Minnesota Extension — Potato Early Blight",
        "management": "Monitor fields regularly and follow locally recommended fungicide programs when necessary."
    },

    {
        "model_class": "Potato___Late_blight",
        "crop": "Potato",
        "disease_name": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "description": "A destructive disease of potato favored by cool, wet conditions.",
        "symptoms": "Water-soaked or dark lesions on leaves that rapidly enlarge; white fungal growth may develop under humid conditions.",
        "prevention": "Use certified disease-free seed, remove volunteer potatoes, monitor weather and fields, and avoid prolonged leaf wetness where possible.",
        "source": "Cornell University — Late Blight of Potato and Tomato",
        "management": "Use an appropriate locally recommended late-blight management program and follow product labels."
    },

    {
        "model_class": "Squash___Powdery_mildew",
        "crop": "Squash",
        "disease_name": "Powdery Mildew",
        "scientific_name": "Podosphaera xanthii / Golovinomyces spp.",
        "description": "A fungal disease that commonly affects cucurbit leaves.",
        "symptoms": "White, powder-like fungal growth on leaf surfaces followed by yellowing and decline of affected foliage.",
        "prevention": "Maintain good air circulation, avoid excessive nitrogen fertilization, remove heavily affected plant material, and use resistant varieties where available.",
        "source": "University of Minnesota Extension — Powdery Mildew of Cucurbits",
        "management": "Monitor plants regularly and follow locally recommended powdery-mildew management practices."
    },

    {
        "model_class": "Strawberry___Leaf_scorch",
        "crop": "Strawberry",
        "disease_name": "Leaf Scorch",
        "scientific_name": "Diplocarpon earlianum",
        "description": "A fungal disease affecting strawberry foliage.",
        "symptoms": "Small purple to reddish spots develop on leaves and may enlarge or merge, giving foliage a scorched appearance.",
        "prevention": "Use healthy planting material, improve air circulation, avoid prolonged leaf wetness, and remove infected plant debris.",
        "source": "University of Minnesota Extension — Strawberry Leaf Diseases",
        "management": "Follow locally recommended strawberry disease-management practices when disease pressure is significant."
    },

        {
        "model_class": "Tomato___Septoria_leaf_spot",
        "crop": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "description": "A fungal disease that primarily affects tomato leaves and can cause significant premature defoliation.",
        "symptoms": "Small circular spots with dark margins and gray or tan centers develop on older leaves. Tiny dark fruiting structures may appear in the centers of lesions.",
        "prevention": "Use good garden sanitation, remove infected plant debris, avoid overhead irrigation, and provide adequate spacing for air circulation.",
        "source": "University of Minnesota Extension — Septoria Leaf Spot",
        "management": "Remove affected foliage where practical and follow locally recommended disease-management practices when disease pressure is significant."
    },

    {
        "model_class": "Tomato___Spider_mites Two-spotted_spider_mite",
        "crop": "Tomato",
        "disease_name": "Two-Spotted Spider Mite",
        "scientific_name": "Tetranychus urticae",
        "description": "A common tomato pest that feeds on plant tissue and can become damaging under hot and dry conditions.",
        "symptoms": "Fine yellow or pale stippling appears on leaves. Severe infestations can cause bronzing, leaf drying, webbing, and premature leaf drop.",
        "prevention": "Monitor plants regularly, reduce dusty conditions, maintain adequate plant moisture, and conserve beneficial predatory mites and insects.",
        "source": "University of Minnesota Extension — Two-Spotted Spider Mites",
        "management": "Monitor mite populations and use appropriate locally recommended management practices while avoiding unnecessary broad-spectrum insecticides that can harm natural enemies."
    },

    {
        "model_class": "Tomato___Target_Spot",
        "crop": "Tomato",
        "disease_name": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "description": "A fungal disease that can affect tomato leaves, stems, and fruit, particularly under warm and humid conditions.",
        "symptoms": "Brown circular lesions with concentric rings can develop on leaves and fruit. Severe leaf infection may result in defoliation.",
        "prevention": "Use good sanitation, remove infected plant debris, provide adequate plant spacing and air circulation, and avoid prolonged leaf wetness.",
        "source": "University of Florida IFAS Extension — Target Spot of Tomato",
        "management": "Monitor crops regularly and follow locally recommended fungicide and cultural management practices when necessary."
    },

    {
        "model_class": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "crop": "Tomato",
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "scientific_name": "Tomato yellow leaf curl virus",
        "description": "A viral disease of tomato transmitted primarily by the silverleaf whitefly.",
        "symptoms": "Plants may show upward curling of leaves, yellowing along leaf margins, reduced leaf size, shortened internodes, and severe stunting. Infected plants may produce few or no marketable fruits.",
        "prevention": "Use healthy planting material, manage whitefly populations, remove infected plants where appropriate, and control weeds that can serve as virus or vector hosts.",
        "source": "University of California Agriculture and Natural Resources — Tomato Yellow Leaf Curl Virus",
        "management": "Manage the whitefly vector and infected plant material according to local integrated pest and disease-management recommendations."
    },

    {
        "model_class": "Tomato___Tomato_mosaic_virus",
        "crop": "Tomato",
        "disease_name": "Tomato Mosaic Virus",
        "scientific_name": "Tomato mosaic virus",
        "description": "A highly transmissible viral disease that can affect tomato plants and reduce plant growth and fruit production.",
        "symptoms": "Leaves may develop mottled light- and dark-green patterns, distortion, or narrowing. Plants may show reduced growth and affected fruit development.",
        "prevention": "Use clean seed and planting material, remove infected plants, control weeds, and maintain good sanitation because the virus can spread mechanically through contaminated hands, tools, and plant material.",
        "source": "University of Minnesota Extension — Tomato Mosaic Virus",
        "management": "Remove infected plants where appropriate and follow strict sanitation practices to reduce mechanical transmission."
    }
]


# ============================================================
# SEED DATABASE
# ============================================================

def seed_database():

    db = SessionLocal()

    inserted = 0
    skipped = 0

    try:

        for item in diseases:

            # ------------------------------------------------
            # Check whether disease already exists
            # ------------------------------------------------

            existing_disease = (
                db.query(Disease)
                .filter(
                    Disease.model_class == item["model_class"]
                )
                .first()
            )

            # ------------------------------------------------
            # Already exists
            # ------------------------------------------------

            if existing_disease:

                print(
                    f"Skipping existing: "
                    f"{item['model_class']}"
                )

                skipped += 1
                continue

            # ------------------------------------------------
            # Create disease record
            # ------------------------------------------------

            disease = Disease(
                crop=item["crop"],
                disease_name=item["disease_name"],
                model_class=item["model_class"],
                scientific_name=item["scientific_name"],
                description=item["description"],
                symptoms=item["symptoms"],
                prevention=item["prevention"]
            )

            db.add(disease)
            db.flush()

            # ------------------------------------------------
            # Create management guideline
            # ------------------------------------------------

            guideline = ManagementGuideline(
                disease_id=disease.id,
                management_type="GENERAL_MANAGEMENT",
                guideline=item["management"],
                source=item["source"],
                last_verified=date.today()
            )

            db.add(guideline)

            inserted += 1

            print(
                f"Inserted: {item['model_class']}"
            )

        # ----------------------------------------------------
        # Commit all changes
        # ----------------------------------------------------

        db.commit()

        print()
        print("=" * 60)
        print("DATABASE SEEDING COMPLETED")
        print("=" * 60)
        print(f"Inserted: {inserted}")
        print(f"Skipped:  {skipped}")
        print(f"Total knowledge-base entries: {len(diseases)}")
        print("=" * 60)

    except Exception as e:

        db.rollback()

        print()
        print("Database seeding failed:")
        print(e)

        raise

    finally:

        db.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    seed_database()