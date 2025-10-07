DECLARE @UNDLP INT = 2071;

SELECT *
FROM (
    SELECT 'ingredients' AS type_, 'UK' AS LANGUAGES,
           CONCAT('(UA) ', REPLACE(REPLACE(CAST(NMAT_UK AS nvarchar(254)), '<', '&lt;'), '>', '&gt;'),
                  _DLP_Z.COMP_UK, _DLP_Z.COND_UK) AS text
    FROM _DLP_Z
    LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
    WHERE _DLP_Z.UNDLP = @UNDLP

    UNION ALL

    SELECT 'ingredients', 'RU',
           CONCAT('(RU) ', CAST(nmat_ru AS nvarchar(254)), _DLP_Z.COMP_RU, _DLP_Z.COND_RU)
    FROM _DLP_Z
    LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
    WHERE _DLP_Z.UNDLP = @UNDLP
	
    UNION ALL

    SELECT 'ingredients', 'EN',
           CONCAT('(EN) ', CAST(nmat_eng AS nvarchar(254)), _DLP_Z.COMP_ENG, _DLP_Z.COND_ENG)
    FROM _DLP_Z
    LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
    WHERE _DLP_Z.UNDLP = @UNDLP

    UNION ALL

    SELECT 'ingredients' type_, sp2.CHAR_SP1 LANGUAGES,
           CONCAT('(',sp2.CHAR_SP1,')',REPLACE(REPLACE(LTRIM(layoutr_.NAME), '<', '&lt;'), '>', '&gt;'), '. ',
                  REPLACE(REPLACE(LTRIM(layoutr_.translat), '<', '&lt;'), '>', '&gt;'),
				  (select (select IMPADDRESS from dtip_ where COUNTRY_AL = replace (rtrim(sp2.CHAR_SP1), ' ', '') and IMPADDRESS  is not null) + (select ' ' + countorig from dtip_ where COUNTRY_AL = replace (rtrim(sp2.CHAR_SP1), ' ', '') and countorig  is not null))) text
    FROM _DLP_Z
    LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
    INNER JOIN layoutr_ ON layoutr_.UNDLP = _DLP_Z.UNDLP
    INNER JOIN SP2 ON SP2.SPR = 'RES' AND SP2.KOD_N = layoutr_.COUNTRY
    WHERE _DLP_Z.UNDLP = @UNDLP
      AND layoutr_.PR_DO = '+'  --and replace (rtrim(sp2.CHAR_SP1), ' ', '')  = 'BG'
		union all
	SELECT	'manufacturing_date' AS type_,
			'' LANGUAGES,
			STUFF((	select text from (
							SELECT distinct dtip_.prdate text
							FROM _DLP_Z 
								LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
								INNER JOIN layoutr_ ON layoutr_.UNDLP = _DLP_Z.UNDLP
								INNER JOIN SP2 ON SP2.SPR = 'RES' AND SP2.KOD_N = layoutr_.COUNTRY
								inner join dtip_ on (COUNTRY_AL = replace (rtrim(sp2.CHAR_SP1), ' ', '') or COUNTRY_AL in  ('RU', 'UA') ) and prdate  is not null
							WHERE _DLP_Z.UNDLP = @UNDLP
							  AND layoutr_.PR_DO = '+'

									union 

							SELECT distinct dtip_.bestbefor text
							FROM _DLP_Z 
								LEFT JOIN ksm ON ksm.kmat = _DLP_Z.KMAT
								INNER JOIN layoutr_ ON layoutr_.UNDLP = _DLP_Z.UNDLP
								INNER JOIN SP2 ON SP2.SPR = 'RES' AND SP2.KOD_N = layoutr_.COUNTRY
								inner join dtip_ on (COUNTRY_AL = replace (rtrim(sp2.CHAR_SP1), ' ', '') or COUNTRY_AL in  ('RU', 'UA') ) and bestbefor  is not null
							WHERE _DLP_Z.UNDLP = @UNDLP
							  AND layoutr_.PR_DO = '+'
							  ) t

              FOR XML PATH('')), 1, 3, '') AS text

) AS combined
FOR JSON PATH;